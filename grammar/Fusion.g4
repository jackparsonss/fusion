grammar Fusion;

file: statement* EOF;

statement:
    declaration
    ;

declaration:
    qualifier ID COLON type EQ expr SEMI
    ;

expr
    : INT #literalInt
    | ID  #variable
    ;

qualifier: CONST | LET;

type: I32;

// symbols
SEMI: ';';
COLON: ':';
EQ: '=';

// comments
LINE_COMMENT: '//' .*? ('\n' | EOF) -> skip;
COMMENT: '/*' .*? '*/' -> skip;

// types
I32: 'i32';

// keywords
CONST: 'const';
LET: 'let';

// Literals
INT: [0-9]+;
ID: [a-zA-Z_] [a-zA-Z0-9_]*;

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
