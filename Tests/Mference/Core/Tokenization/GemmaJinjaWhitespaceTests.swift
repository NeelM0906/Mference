import Jinja
import Testing

@Suite("Jinja whitespace around Gemma literal braces")
struct GemmaJinjaWhitespaceTests {
    @Test func pythonNullOutputIsScopedAndPreservesNullTests() throws {
        let template = try Template("{% macro value(x) %}{{ x }}{% endmacro %}{{ value(none) }}|{{ missing }}|{{ none is none }}")
        let environment = Environment()
        environment.policies.nullOutput = "None"
        #expect(try template.render([:], environment: environment) == "None||true")
        #expect(try template.render([:]) == "||true")
        let conditional = try Template("{{ 'x' if false }}|{{ 'x' if false else none }}|{{ ('x' if false) is undefined }}")
        #expect(try conditional.render([:], environment: environment) == "|None|true")
        #expect(try conditional.render([:]) == "||true")
    }

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
