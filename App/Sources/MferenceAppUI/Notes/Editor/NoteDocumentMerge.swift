import MferenceAppCore

/// A block-level three-way merge: the person's unsaved edits (`ours`) and a
/// write from elsewhere (`theirs`, e.g. a meeting summary or a marker) made
/// to the same stored body (`base`). Nothing either side added is lost:
/// blocks removed by one side and kept by the other go; a block both sides
/// changed appears in both versions.
enum NoteDocumentMerge {
    static func merge(base: [NoteDocument.Block], ours: [NoteDocument.Block],
                      theirs: [NoteDocument.Block]) -> [NoteDocument.Block] {
        if ours == base { return theirs }
        if theirs == base || theirs == ours { return ours }
        let toTheirs = matches(base, theirs)
        // Walk base against ours: kept, removed (or replaced) and added blocks.
        let ourDifference = ours.difference(from: base)
        var removedByOurs = Set<Int>(), addedByOurs = Set<Int>()
        for change in ourDifference {
            switch change {
            case .remove(let offset, _, _): removedByOurs.insert(offset)
            case .insert(let offset, _, _): addedByOurs.insert(offset)
            }
        }
        var keep = Array(repeating: true, count: theirs.count)
        // Blocks the person added go after the last base block before them
        // that theirs still has; a replaced block's new version takes the
        // old one's place in theirs (-1: at the very start).
        var inserted: [Int: [NoteDocument.Block]] = [:]
        var anchor = -1
        var b = 0, o = 0
        while b < base.count || o < ours.count {
            if b < base.count, removedByOurs.contains(b) {
                if let index = toTheirs[b] {
                    keep[index] = false
                    anchor = index
                }
                b += 1
            } else if o < ours.count, addedByOurs.contains(o) {
                inserted[anchor, default: []].append(ours[o])
                o += 1
            } else {
                if b < base.count, let index = toTheirs[b] { anchor = index }
                b += 1
                o += 1
            }
        }
        var result = inserted[-1] ?? []
        for (index, block) in theirs.enumerated() {
            if keep[index] { result.append(block) }
            result.append(contentsOf: inserted[index] ?? [])
        }
        return result
    }

    /// Unchanged blocks: index in `a` → index in `b`.
    static func matches(_ a: [NoteDocument.Block], _ b: [NoteDocument.Block]) -> [Int: Int] {
        let difference = b.difference(from: a)
        var removed = Set<Int>(), added = Set<Int>()
        for change in difference {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): added.insert(offset)
            }
        }
        let kept = a.indices.filter { !removed.contains($0) }
        let keptInB = b.indices.filter { !added.contains($0) }
        return Dictionary(uniqueKeysWithValues: zip(kept, keptInB))
    }
}
