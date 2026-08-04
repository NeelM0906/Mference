import Foundation

/// Seed sidebar data ported from the template's `src/utils/mock-chats.ts`.
public enum MockChats {
    private static let entries: [(title: String, daysAgo: Int, starred: Bool)] = [
        ("Expo Job offer", 5, false),
        ("Existing tools for iOS app tech stack detection", 5, false),
        ("Headless iOS simulator gateway for concurrent testing", 7, false),
        ("Top three.js projects", 7, true),
        ("Austin magician review", 7, false),
        ("Expo agent GitHub bot description", 14, false),
        ("Building an iMessage bot with Claude", 14, true),
        ("Conditional HMR disabling in web frameworks", 14, false),
        ("Optimizing parallel git config queries", 14, false),
        ("Choosing between Tailwind and StyleX", 21, false),
        ("Structuring messages and timelines", 28, false),
        ("SVG morphing animation between shapes", 28, false),
        ("Expo navigation patterns", 30, false),
        ("Debugging Expo CLI", 35, false),
    ]

    public static func seed(now: Date = Date()) -> [Chat] {
        entries.map { entry in
            Chat(
                title: entry.title,
                lastActivity: now.addingTimeInterval(TimeInterval(-entry.daysAgo * 86_400)),
                starred: entry.starred)
        }
    }
}
