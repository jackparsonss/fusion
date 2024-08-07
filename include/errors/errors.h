#pragma once
#include <ostream>
#include <sstream>

class CompileTimeException : public std::exception {
   protected:
    std::string msg;

   public:
    const char* what() const noexcept override { return msg.c_str(); }
};

class RunTimeException : public std::exception {
   protected:
    std::string msg;

   public:
    const char* what() const noexcept override { return msg.c_str(); }
};

#define DEF_COMPILE_TIME_EXCEPTION(NAME)                               \
    class NAME : public CompileTimeException {                         \
       public:                                                         \
        NAME(unsigned line, const std::string& description) {          \
            std::stringstream buf;                                     \
            buf << #NAME << " on Line " << line << ": " << description \
                << std::endl;                                          \
            msg = buf.str();                                           \
        }                                                              \
    }

#define DEF_RUNTIME_TIME_EXCEPTION(NAME)                      \
    class NAME : public CompileTimeException {                \
       public:                                                \
        NAME(const std::string& description) {                \
            std::stringstream buf;                            \
            buf << #NAME << ": " << description << std::endl; \
            msg = buf.str();                                  \
        }                                                     \
    }

DEF_COMPILE_TIME_EXCEPTION(MainError);

DEF_COMPILE_TIME_EXCEPTION(SymbolError);

DEF_COMPILE_TIME_EXCEPTION(LexerError);

DEF_COMPILE_TIME_EXCEPTION(SyntaxError);

DEF_COMPILE_TIME_EXCEPTION(TypeError);

DEF_COMPILE_TIME_EXCEPTION(AssignError);

DEF_RUNTIME_TIME_EXCEPTION(BackendError);
