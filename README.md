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
- **i32**: 32-bit integer

### Functions
```
fn foo(let x: i32): i32 {
    print(x);

    return x + 1;
}

fn main(): i32 {
    foo(5);
    return 0;
}
```

#### Builtin Functions
##### main
Must be defined by the user, is the entry point to the fusion program
```
fn main(): i32 {
    // do stuff
    return 0;
}
```

##### print
Prints the argument passed to it to stdout
```
print(5);
```
