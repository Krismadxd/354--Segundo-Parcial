-- Declaración de variables
DECLARE
    @PALABRA1 VARCHAR(20),
    @PALABRA2 VARCHAR(20),
    @LONGITUD INT,
    @CONTADOR INT,
    @RESULTADO INT;

-- Definición de las palabras a comparar
SET @PALABRA1 = 'INFORMATICA';
SET @PALABRA2 = 'Informtaica';  

-- Calcular la longitud de la palabra 1
SET @LONGITUD = LEN(@PALABRA1);
-- Inicializar contador y resultado
SET @CONTADOR = 1;
SET @RESULTADO = 0;

-- Imprimir las palabras que se están comparando
PRINT 'Palabra 1: ' + @PALABRA1;
PRINT 'Palabra 2: ' + @PALABRA2;
PRINT '';

-- Imprimir encabezado de la matriz de comparación
PRINT 'Matriz de Comparación:';
PRINT '----------------------';

-- Imprimir las letras y sus comparaciones
WHILE @CONTADOR <= @LONGITUD
BEGIN
    -- Imprimir cada letra de las palabras en líneas separadas para comparar visualmente
    PRINT SUBSTRING(@PALABRA1, @CONTADOR, 1) + '   ' + SUBSTRING(@PALABRA2, @CONTADOR, 1);
    SET @CONTADOR = @CONTADOR + 1;
END

-- Comparar letra por letra y contar coincidencias
SET @CONTADOR = 1;
WHILE @CONTADOR <= @LONGITUD
BEGIN
    -- Si las letras en la misma posición son iguales, aumentar el contador de coincidencias
    IF SUBSTRING(@PALABRA1, @CONTADOR, 1) = SUBSTRING(@PALABRA2, @CONTADOR, 1)
    BEGIN
        SET @RESULTADO = @RESULTADO + 1;
    END

    SET @CONTADOR = @CONTADOR + 1;
END

-- Imprimir el resultado final de coincidencias
PRINT '';
PRINT 'Número de letras coincidentes entre ' + @PALABRA1 + ' y ' + @PALABRA2 + ': ' + CAST(@RESULTADO AS VARCHAR(10));
