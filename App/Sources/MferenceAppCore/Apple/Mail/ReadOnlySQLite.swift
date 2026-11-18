import Foundation
import SQLite3

/// A minimal read-only SQLite connection (`SQLITE_OPEN_READONLY`, `mode=ro`)
/// for the private copy of Mail's Envelope Index.
final class ReadOnlySQLite {
    enum Value: Sendable, Hashable {
        case integer(Int64)
        case real(Double)
        case text(String)
        case null

        var int: Int64? { if case .integer(let value) = self { value } else { nil } }
        var double: Double? {
            switch self {
            case .real(let value): value
            case .integer(let value): Double(value)
            default: nil
            }
        }
        var string: String? { if case .text(let value) = self { value } else { nil } }
    }

    struct Failure: Error, Sendable, CustomStringConvertible {
        var code: Int32
        var message: String
        var description: String { "SQLite \(code): \(message)" }
    }

    private var handle: OpaquePointer?

    /// Opens `url` read-only and immutable (no locks, no -shm, no writes).
    /// `AppleMailSource` only ever passes its private, checkpointed copy of
    /// Mail's index here, never Mail's live file.
    init(url: URL) throws {
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false) ?? URLComponents()
        components.scheme = "file"
        components.queryItems = [URLQueryItem(name: "mode", value: "ro"), URLQueryItem(name: "immutable", value: "1")]
        let uri = components.string ?? "file:\(url.path)?mode=ro"
        let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_URI | SQLITE_OPEN_NOMUTEX
        let code = sqlite3_open_v2(uri, &handle, flags, nil)
        guard code == SQLITE_OK else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close(handle)
            handle = nil
            throw Failure(code: code, message: message)
        }
        sqlite3_busy_timeout(handle, 2_000)
    }

    deinit {
        sqlite3_close(handle)
    }

    /// Folds a private copy's write-ahead log into the copy, so an immutable
    /// read of it still sees the newest mail. Only ever called on the copy in
    /// `AppleMailSource`'s private folder.
    static func checkpointPrivateCopy(at url: URL) throws {
        var connection: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX
        guard sqlite3_open_v2(url.path, &connection, flags, nil) == SQLITE_OK else {
            let message = connection.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close(connection)
            throw Failure(code: SQLITE_CANTOPEN, message: message)
        }
        defer { sqlite3_close(connection) }
        sqlite3_busy_timeout(connection, 2_000)
        // Not in WAL mode, or nothing to fold: both fine.
        _ = sqlite3_exec(connection, "PRAGMA wal_checkpoint(TRUNCATE)", nil, nil, nil)
    }

    /// Rows as column-name dictionaries. `parameters` bind to `?` in order.
    func query(_ sql: String, _ parameters: [Value] = []) throws -> [[String: Value]] {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else { throw lastError() }
        defer { sqlite3_finalize(statement) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        for (index, parameter) in parameters.enumerated() {
            let position = Int32(index + 1)
            switch parameter {
            case .integer(let value): sqlite3_bind_int64(statement, position, value)
            case .real(let value): sqlite3_bind_double(statement, position, value)
            case .text(let value): sqlite3_bind_text(statement, position, value, -1, transient)
            case .null: sqlite3_bind_null(statement, position)
            }
        }
        var rows: [[String: Value]] = []
        while true {
            let step = sqlite3_step(statement)
            if step == SQLITE_DONE { break }
            guard step == SQLITE_ROW else { throw lastError() }
            var row: [String: Value] = [:]
            for column in 0..<sqlite3_column_count(statement) {
                let name = String(cString: sqlite3_column_name(statement, column))
                switch sqlite3_column_type(statement, column) {
                case SQLITE_INTEGER: row[name] = .integer(sqlite3_column_int64(statement, column))
                case SQLITE_FLOAT: row[name] = .real(sqlite3_column_double(statement, column))
                case SQLITE_TEXT: row[name] = .text(String(cString: sqlite3_column_text(statement, column)))
                default: row[name] = .null
                }
            }
            rows.append(row)
        }
        return rows
    }

    /// Column names of `table`, empty when the table does not exist.
    func columns(of table: String) throws -> Set<String> {
        Set(try query("SELECT name FROM pragma_table_info(?)", [.text(table)]).compactMap { $0["name"]?.string })
    }

    private func lastError() -> Failure {
        Failure(code: sqlite3_errcode(handle), message: String(cString: sqlite3_errmsg(handle)))
    }
}
