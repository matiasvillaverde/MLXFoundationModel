#include "CXGrammarBridge.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <xgrammar/xgrammar.h>

struct CXGGrammarCompiler {
    explicit CXGGrammarCompiler(xgrammar::TokenizerInfo tokenizer_info)
        : tokenizer_info(std::move(tokenizer_info)) {}

    xgrammar::TokenizerInfo tokenizer_info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
};

struct CXGGrammarMatcher {
    CXGGrammarMatcher(int32_t vocab_size, std::unique_ptr<xgrammar::GrammarMatcher> matcher)
        : vocab_size(vocab_size), matcher(std::move(matcher)) {}

    int32_t vocab_size;
    std::unique_ptr<xgrammar::GrammarMatcher> matcher;
};

namespace {

char *copy_string(const std::string &message) {
    auto *result = static_cast<char *>(std::malloc(message.size() + 1));
    if (result == nullptr) {
        return nullptr;
    }
    std::memcpy(result, message.c_str(), message.size() + 1);
    return result;
}

void set_error(char **error_message, const std::string &message) {
    if (error_message != nullptr) {
        *error_message = copy_string(message);
    }
}

std::vector<std::string> make_vocab(const char *const *encoded_vocab, int32_t vocab_count) {
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>(vocab_count));
    for (int32_t index = 0; index < vocab_count; ++index) {
        result.emplace_back(encoded_vocab[index] == nullptr ? "" : encoded_vocab[index]);
    }
    return result;
}

std::vector<int32_t> make_stop_tokens(const int32_t *stop_token_ids, int32_t stop_token_count) {
    std::vector<int32_t> result;
    result.reserve(static_cast<std::size_t>(stop_token_count));
    for (int32_t index = 0; index < stop_token_count; ++index) {
        result.push_back(stop_token_ids[index]);
    }
    return result;
}

std::string enriched_metadata(
    const std::string &tokenizer_json,
    int32_t vocab_count,
    const std::vector<int32_t> &stop_tokens
) {
    std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(tokenizer_json);
    if (metadata.empty() || metadata.back() != '}') {
        throw std::runtime_error("XGrammar tokenizer metadata is malformed");
    }

    std::ostringstream stream;
    stream << metadata.substr(0, metadata.size() - 1);
    stream << ",\"vocab_size\":" << vocab_count;
    stream << ",\"stop_token_ids\":[";
    for (std::size_t index = 0; index < stop_tokens.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << stop_tokens[index];
    }
    stream << "]}";
    return stream.str();
}

DLTensor make_bitmask_tensor(int32_t *bitmask, int64_t *shape) {
    DLTensor tensor;
    tensor.data = bitmask;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
}

CXGGrammarMatcher *make_matcher(xgrammar::CompiledGrammar compiled_grammar) {
    int32_t vocab_size = compiled_grammar.GetTokenizerInfo().GetVocabSize();
    auto matcher = std::make_unique<xgrammar::GrammarMatcher>(
        compiled_grammar,
        std::nullopt,
        false,
        -1
    );
    return new CXGGrammarMatcher(vocab_size, std::move(matcher));
}

template <typename Operation>
auto run_bridge(Operation operation, char **error_message) -> decltype(operation()) {
    try {
        if (error_message != nullptr) {
            *error_message = nullptr;
        }
        return operation();
    } catch (const std::exception &error) {
        set_error(error_message, error.what());
        return {};
    } catch (...) {
        set_error(error_message, "Unknown XGrammar bridge error");
        return {};
    }
}

}  // namespace

CXGGrammarCompiler *cxg_compiler_create(
    const char *const *encoded_vocab,
    int32_t vocab_count,
    const int32_t *stop_token_ids,
    int32_t stop_token_count,
    const char *tokenizer_json,
    char **error_message
) {
    return run_bridge([&]() -> CXGGrammarCompiler * {
        if (encoded_vocab == nullptr || tokenizer_json == nullptr || vocab_count <= 0) {
            throw std::invalid_argument("Invalid tokenizer vocabulary");
        }
        std::vector<std::string> vocab = make_vocab(encoded_vocab, vocab_count);
        std::vector<int32_t> stop_tokens = make_stop_tokens(stop_token_ids, stop_token_count);
        std::string metadata = enriched_metadata(tokenizer_json, vocab_count, stop_tokens);

        auto tokenizer_info = xgrammar::TokenizerInfo::FromVocabAndMetadata(vocab, metadata);
        auto *handle = new CXGGrammarCompiler(std::move(tokenizer_info));
        handle->compiler = std::make_unique<xgrammar::GrammarCompiler>(
            handle->tokenizer_info,
            8,
            true
        );
        return handle;
    }, error_message);
}

void cxg_compiler_destroy(CXGGrammarCompiler *compiler) {
    delete compiler;
}

CXGGrammarMatcher *cxg_compiler_compile_json_schema(
    CXGGrammarCompiler *compiler,
    const char *schema,
    bool strict_mode,
    char **error_message
) {
    return run_bridge([&]() -> CXGGrammarMatcher * {
        if (compiler == nullptr || schema == nullptr) {
            throw std::invalid_argument("Invalid JSON schema compiler input");
        }
        return make_matcher(compiler->compiler->CompileJSONSchema(
            schema,
            true,
            std::nullopt,
            std::nullopt,
            strict_mode
        ));
    }, error_message);
}

CXGGrammarMatcher *cxg_compiler_compile_builtin_json(
    CXGGrammarCompiler *compiler,
    char **error_message
) {
    return run_bridge([&]() -> CXGGrammarMatcher * {
        if (compiler == nullptr) {
            throw std::invalid_argument("Invalid JSON compiler input");
        }
        return make_matcher(compiler->compiler->CompileBuiltinJSONGrammar());
    }, error_message);
}

CXGGrammarMatcher *cxg_compiler_compile_ebnf(
    CXGGrammarCompiler *compiler,
    const char *grammar,
    const char *root_rule_name,
    char **error_message
) {
    return run_bridge([&]() -> CXGGrammarMatcher * {
        if (compiler == nullptr || grammar == nullptr || root_rule_name == nullptr) {
            throw std::invalid_argument("Invalid EBNF compiler input");
        }
        return make_matcher(compiler->compiler->CompileGrammar(grammar, root_rule_name));
    }, error_message);
}

CXGGrammarMatcher *cxg_compiler_compile_regex(
    CXGGrammarCompiler *compiler,
    const char *regex,
    char **error_message
) {
    return run_bridge([&]() -> CXGGrammarMatcher * {
        if (compiler == nullptr || regex == nullptr) {
            throw std::invalid_argument("Invalid regex compiler input");
        }
        return make_matcher(compiler->compiler->CompileRegex(regex));
    }, error_message);
}

void cxg_matcher_destroy(CXGGrammarMatcher *matcher) {
    delete matcher;
}

int32_t cxg_matcher_vocab_size(const CXGGrammarMatcher *matcher) {
    return matcher == nullptr ? 0 : matcher->vocab_size;
}

int32_t cxg_matcher_bitmask_size(const CXGGrammarMatcher *matcher) {
    if (matcher == nullptr) {
        return 0;
    }
    return xgrammar::GetBitmaskSize(matcher->vocab_size);
}

bool cxg_matcher_fill_next_token_bitmask(
    CXGGrammarMatcher *matcher,
    int32_t *bitmask,
    int32_t bitmask_count,
    char **error_message
) {
    return run_bridge([&]() -> bool {
        if (matcher == nullptr || bitmask == nullptr) {
            throw std::invalid_argument("Invalid grammar matcher input");
        }
        if (bitmask_count != xgrammar::GetBitmaskSize(matcher->vocab_size)) {
            throw std::invalid_argument("Invalid grammar bitmask size");
        }
        int64_t shape[] = {bitmask_count};
        DLTensor tensor = make_bitmask_tensor(bitmask, shape);
        return matcher->matcher->FillNextTokenBitmask(&tensor);
    }, error_message);
}

bool cxg_matcher_accept_token(
    CXGGrammarMatcher *matcher,
    int32_t token_id,
    char **error_message
) {
    return run_bridge([&]() -> bool {
        if (matcher == nullptr) {
            throw std::invalid_argument("Invalid grammar matcher input");
        }
        return matcher->matcher->AcceptToken(token_id);
    }, error_message);
}

bool cxg_matcher_is_completed(const CXGGrammarMatcher *matcher) {
    return matcher != nullptr && matcher->matcher->IsCompleted();
}

bool cxg_matcher_is_terminated(const CXGGrammarMatcher *matcher) {
    return matcher != nullptr && matcher->matcher->IsTerminated();
}

void cxg_free_string(char *string) {
    std::free(string);
}
