#include "shared/type.h"

Type::Type(NativeType type) {
    this->base = type;
}

NativeType Type::get_base() const {
    return this->base;
}
