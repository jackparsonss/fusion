# Fusion Lang
A toy language to continue learning antlr4 and mlir/llvm and see where I go

## Setting up environment
If you are on macos then I have made a simple install script which will install all requirements/dependencies needed to develop with antlr and mlir, located [here](https://github.com/jackparsonss/fusion/blob/main/scipts/setup_macos.bash)

## Building
```bash
mkdir build
cd build
cmake ..
make
```

## Running
```bash
cd bin
./fuse <filename>.fuse
```

## Documentation
### Declarations
The `let` keyword signifies that the variable can be later changed.
The `const` keyword signifies that the variable cannot be later changed;
```
let x: i32 = 5;
const y: i32 = x;
```

### Types
#### i32
A 32-bit integer
