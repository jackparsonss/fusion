; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define i32 @main() {
  %1 = call i32 @_rxijpuiczshoairydebmgygigztbezsngzycycxeelsjlnvcpoxadeleornj(i32 5)
  ret i32 0
}

define i32 @_rxijpuiczshoairydebmgygigztbezsngzycycxeelsjlnvcpoxadeleornj(i32 %0) {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  call void @print(i32 %3)
  ret i32 0
}

declare void @print_integer(i32)

define void @print(i32 %0) {
  call void @print_integer(i32 %0)
  ret void
}
