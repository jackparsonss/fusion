grammar Fusion;

file: .*? EOF;

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
