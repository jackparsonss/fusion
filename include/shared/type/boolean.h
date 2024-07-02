#pragma once

#include "shared/type/type.h"

class Boolean : public Type {
   public:
    Boolean();
    std::string get_specifier() const override;
    mlir::Type get_mlir() const override;
};
