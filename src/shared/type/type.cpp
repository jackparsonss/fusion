#include <stdexcept>

#include "shared/context.h"
#include "shared/type/type.h"

Type::Type(std::string name) {
    this->name = name;
}

bool Type::operator==(const Type rhs) const {
    return name == rhs.name || rhs.name == "any";
}

std::string Type::get_name() const {
    return this->name;
}

std::string Type::get_specifier() const {
    throw std::runtime_error("type " + name + " is not printable");
}

mlir::Type Type::get_mlir() const {
    throw std::runtime_error("invalid mlir type found " + name);
}

bool Type::is_numeric() const {
    return false;
}

mlir::Type Type::get_pointer() {
    return mlir::LLVM::LLVMPointerType::get(get_mlir());
}

Any::Any() : Type("any") {}

std::string Any::get_specifier() const {
    return "any";
}

None::None() : Type("none") {}

std::string None::get_specifier() const {
    return "none";
}

mlir::Type None::get_mlir() const {
    return mlir::LLVM::LLVMVoidType::get(&ctx::context);
}

Unset::Unset() : Type("unset") {}

std::string Unset::get_specifier() const {
    return "unset";
}
