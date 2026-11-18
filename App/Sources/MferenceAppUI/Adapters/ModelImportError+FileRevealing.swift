import MferenceAppCore

/// The library's import alert offers Show in Finder for the file that kept
/// a model file from being added (U1's `FileRevealingError`).
extension ModelImportError: FileRevealingError {}
