grammar Fusion;

file: statement* EOF;

statement
    : function
    | call SEMI
    | declaration
    | block
    | return
    ;

declaration:
    variable EQ expr SEMI
    ;

block: L_CURLY statement* R_CURLY;

function: FUNCTION ID L_PAREN variable* R_PAREN COLON type block;

variable: qualifier ID COLON type;

call: ID L_PAREN expr* R_PAREN;

return: RETURN expr SEMI;

expr
    : INT   #literalInt
    | ID    #identifier
    | call  #callExpr 
    ;

qualifier: CONST | LET;

type: I32;

// keywords
RETURN: 'return';
FUNCTION: 'fn';
CONST: 'const';
LET: 'let';

// symbols
SEMI: ';';
COLON: ':';
EQ: '=';
L_PAREN: '(';
R_PAREN: ')';
L_CURLY: '{';
R_CURLY: '}';

// comments
LINE_COMMENT: '//' .*? ('\n' | EOF) -> skip;
COMMENT: '/*' .*? '*/' -> skip;

// types
I32: 'i32';

// literals
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;

// skip whitespace
WS : [ \t\r\n]+ -> skip ;
