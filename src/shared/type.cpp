#include <stdexcept>

#include "shared/context.h"
#include "shared/type.h"

const Type Type::ch = Type("ch");
const Type Type::i32 = Type("i32");
const Type Type::f32 = Type("f32");
const Type Type::none = Type("none");
const Type Type::unset = Type("unset");

Type::Type(std::string name) {
    this->name = name;
}

bool Type::operator==(const Type rhs) const {
    return this->name == rhs.name;
}

std::string Type::get_name() const {
    return this->name;
}

mlir::Type Type::get_mlir() const {
    if (*this == Type::ch) {
        return ctx::builder->getI8Type();
    }

    if (*this == Type::i32) {
        return ctx::builder->getI32Type();
    }

    if (*this == Type::f32) {
        return ctx::builder->getF32Type();
    }

    if (*this == Type::none) {
        return mlir::LLVM::LLVMVoidType::get(&ctx::context);
    }

    throw std::runtime_error("invalid mlir type found");
}
