#include "shared/type/character.h"
#include "shared/context.h"

Character::Character() : Type("ch") {}

std::string Character::get_specifier() const {
    return "%c";
}

mlir::Type Character::get_mlir() const {
    return ctx::builder->getI8Type();
}
