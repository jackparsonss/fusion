#include "shared/type.h"

Type::Type(NativeType type) {
    this->base = type;
}

NativeType Type::get_base() const {
    return this->base;
}

std::string Type::to_string() {
    switch (this->base) {
        case NativeType::Int32:
            return "i32";
    }

    throw std::runtime_error("found type without a native type");
}
