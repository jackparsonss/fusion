#pragma once

#include <string>
#include "mlir/IR/Types.h"
#include "shared/type/type.h"

class Character : public Type {
   public:
    Character();
    std::string get_specifier() const override;
    mlir::Type get_mlir() const override;
};
