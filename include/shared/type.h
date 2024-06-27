#pragma once

#include <memory>
#include <string>
#include "mlir/IR/Types.h"

class Type {
   private:
    std::string name;

   public:
    explicit Type(std::string name);
    std::string get_name() const;
    std::string get_specifier() const;
    mlir::Type get_mlir() const;
    mlir::Type get_pointer();

    bool operator==(const Type rhs) const;

    static const Type ch;
    static const Type any;
    static const Type i32;
    static const Type f32;
    static const Type none;
    static const Type unset;
    static const Type t_bool;
};

typedef std::shared_ptr<Type> TypePtr;
