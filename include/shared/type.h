#pragma once

#include <memory>

enum class NativeType {
    Int32,
};

class Type {
   private:
    NativeType base;

   public:
    explicit Type(NativeType type);
    NativeType get_base() const;
};

typedef std::shared_ptr<Type> TypePtr;
