#pragma once

#include <memory>
#include <string>

enum class NativeType {
    Int32,
};

class Type {
   private:
    NativeType base;

   public:
    explicit Type(NativeType type);
    NativeType get_base() const;
    std::string to_string();
};

typedef std::shared_ptr<Type> TypePtr;
