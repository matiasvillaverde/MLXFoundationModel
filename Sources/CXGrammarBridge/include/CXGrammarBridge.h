#ifndef CX_GRAMMAR_BRIDGE_H
#define CX_GRAMMAR_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CXGGrammarCompiler CXGGrammarCompiler;
typedef struct CXGGrammarMatcher CXGGrammarMatcher;

CXGGrammarCompiler *cxg_compiler_create(
    const char *const *encoded_vocab,
    int32_t vocab_count,
    const int32_t *stop_token_ids,
    int32_t stop_token_count,
    const char *tokenizer_json,
    char **error_message
);

void cxg_compiler_destroy(CXGGrammarCompiler *compiler);

CXGGrammarMatcher *cxg_compiler_compile_json_schema(
    CXGGrammarCompiler *compiler,
    const char *schema,
    bool strict_mode,
    char **error_message
);

CXGGrammarMatcher *cxg_compiler_compile_builtin_json(
    CXGGrammarCompiler *compiler,
    char **error_message
);

CXGGrammarMatcher *cxg_compiler_compile_ebnf(
    CXGGrammarCompiler *compiler,
    const char *grammar,
    const char *root_rule_name,
    char **error_message
);

CXGGrammarMatcher *cxg_compiler_compile_structural_tag(
    CXGGrammarCompiler *compiler,
    const char *structural_tag_json,
    char **error_message
);

CXGGrammarMatcher *cxg_compiler_compile_regex(
    CXGGrammarCompiler *compiler,
    const char *regex,
    char **error_message
);

void cxg_matcher_destroy(CXGGrammarMatcher *matcher);

int32_t cxg_matcher_vocab_size(const CXGGrammarMatcher *matcher);

int32_t cxg_matcher_bitmask_size(const CXGGrammarMatcher *matcher);

bool cxg_matcher_fill_next_token_bitmask(
    CXGGrammarMatcher *matcher,
    int32_t *bitmask,
    int32_t bitmask_count,
    char **error_message
);

bool cxg_matcher_batch_fill_next_token_bitmask(
    CXGGrammarMatcher *const *matchers,
    int32_t matcher_count,
    int32_t *bitmask,
    int32_t batch_count,
    int32_t bitmask_count,
    char **error_message
);

bool cxg_matcher_accept_token(
    CXGGrammarMatcher *matcher,
    int32_t token_id,
    char **error_message
);

bool cxg_matcher_is_completed(const CXGGrammarMatcher *matcher);

bool cxg_matcher_is_terminated(const CXGGrammarMatcher *matcher);

int32_t cxg_bitmask_count_accepted(
    const int32_t *bitmask,
    int32_t bitmask_count,
    int32_t vocab_size
);

int32_t cxg_bitmask_fill_token_ids(
    const int32_t *bitmask,
    int32_t bitmask_count,
    int32_t vocab_size,
    bool accepted_state,
    int32_t *output,
    int32_t output_count
);

void cxg_free_string(char *string);

#ifdef __cplusplus
}
#endif

#endif
