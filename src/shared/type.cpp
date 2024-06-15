#include "shared/type.h"

const Type Type::i32 = Type("i32");
const Type Type::none = Type("none");
const Type Type::unset = Type("unset");

Type::Type(std::string name) {
    this->name = name;
}

bool Type::operator==(Type rhs) const {
    return this->name == rhs.name;
}

std::string Type::get_name() const {
    return this->name;
}
