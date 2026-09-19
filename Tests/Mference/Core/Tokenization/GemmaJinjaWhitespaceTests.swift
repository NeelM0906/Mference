import Jinja
import Testing

@Suite("Jinja whitespace around Gemma literal braces")
struct GemmaJinjaWhitespaceTests {
    @Test func literalBracesDoNotAcquireProtectiveSpaces() throws {
        let source = "parameters:{\n{%- if true -%}properties:{ {{- 'value' -}} }{%- endif -%}"
        #expect(try Template(source).render([:]) == "parameters:{properties:{value}")
        #expect(try Template("a{ \n{#- note -#}\n b").render([:]) == "a{b")
    }

    @Test func quotedDelimitersAreData() throws {
        #expect(try Template("{{ 'a -}} b' }}").render([:]) == "a -}} b")
        #expect(try Template("{{ 'a {%- b' }}").render([:]) == "a {%- b")
    }
}
