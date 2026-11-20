import Foundation
import SQLite3

/// A small wrapper over the system SQLite (which ships with FTS5). Not
/// thread-safe: `FindIndex` owns one connection inside its actor.
final class FindSQLite {
    enum Failure: Error, CustomStringConvertible {
        case open(String)
        case sql(String, String)

        var description: String {
            switch self {
            case .open(let message): "Find could not open its index: \(message)"
            case .sql(let message, let statement): "Find index error: \(message) (\(statement.prefix(80)))"
            }
        }
    }

    private var handle: OpaquePointer?
    private var statements: [String: OpaquePointer] = [:]

    /// `nil` path opens an in-memory database (tests).
    init(path: String?) throws {
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX
        guard sqlite3_open_v2(path ?? ":memory:", &handle, flags, nil) == SQLITE_OK else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(handle)
            throw Failure.open(message)
        }
        sqlite3_busy_timeout(handle, 5_000)
    }

    deinit { close() }

    func close() {
        for statement in statements.values { sqlite3_finalize(statement) }
        statements = [:]
        if let handle { sqlite3_close(handle) }
        handle = nil
    }

    func execute(_ sql: String) throws {
        var error: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(handle, sql, nil, nil, &error) == SQLITE_OK else {
            let message = error.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(error)
            throw Failure.sql(message, sql)
        }
    }

    var lastInsertRowID: Int64 { sqlite3_last_insert_rowid(handle) }
    var changes: Int { Int(sqlite3_changes(handle)) }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE")
        do {
            let value = try body()
            try execute("COMMIT")
            return value
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    /// Runs `sql` with `bindings`, calling `row` for each result row.
    func query(_ sql: String, _ bindings: [Value] = [], row: (Row) throws -> Void = { _ in }) throws {
        let statement = try prepared(sql)
        defer {
            sqlite3_reset(statement)
            sqlite3_clear_bindings(statement)
        }
        for (index, value) in bindings.enumerated() {
            let position = Int32(index + 1)
            switch value {
            case .null: sqlite3_bind_null(statement, position)
            case .int(let int): sqlite3_bind_int64(statement, position, int)
            case .double(let double): sqlite3_bind_double(statement, position, double)
            case .text(let text): sqlite3_bind_text(statement, position, text, -1, Self.transient)
            case .blob(let data):
                _ = data.withUnsafeBytes { bytes in
                    sqlite3_bind_blob(statement, position, bytes.baseAddress, Int32(bytes.count), Self.transient)
                }
            }
        }
        while true {
            let status = sqlite3_step(statement)
            if status == SQLITE_DONE { break }
            guard status == SQLITE_ROW else {
                throw Failure.sql(String(cString: sqlite3_errmsg(handle)), sql)
            }
            try row(Row(statement: statement))
        }
    }

    func run(_ sql: String, _ bindings: [Value] = []) throws {
        try query(sql, bindings)
    }

    func scalarInt(_ sql: String, _ bindings: [Value] = []) throws -> Int64? {
        var result: Int64?
        try query(sql, bindings) { result = $0.int(0) }
        return result
    }

    private func prepared(_ sql: String) throws -> OpaquePointer {
        if let statement = statements[sql] { return statement }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK, let statement else {
            throw Failure.sql(String(cString: sqlite3_errmsg(handle)), sql)
        }
        statements[sql] = statement
        return statement
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    enum Value {
        case null
        case int(Int64)
        case double(Double)
        case text(String)
        case blob(Data)
    }

    struct Row {
        let statement: OpaquePointer

        func int(_ column: Int32) -> Int64 { sqlite3_column_int64(statement, column) }
        func double(_ column: Int32) -> Double { sqlite3_column_double(statement, column) }
        func isNull(_ column: Int32) -> Bool { sqlite3_column_type(statement, column) == SQLITE_NULL }

        func text(_ column: Int32) -> String {
            guard let pointer = sqlite3_column_text(statement, column) else { return "" }
            return String(cString: pointer)
        }

        /// Calls `body` with the blob's bytes, valid only inside `body`.
        func withBlob<T>(_ column: Int32, _ body: (UnsafeRawBufferPointer) -> T) -> T {
            let count = Int(sqlite3_column_bytes(statement, column))
            guard let pointer = sqlite3_column_blob(statement, column), count > 0 else {
                return body(UnsafeRawBufferPointer(start: nil, count: 0))
            }
            return body(UnsafeRawBufferPointer(start: pointer, count: count))
        }
    }
}
