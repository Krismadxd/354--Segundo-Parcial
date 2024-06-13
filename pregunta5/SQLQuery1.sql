DECLARE
    @PALABRA1 VARCHAR(20),
    @PALABRA2 VARCHAR(20),
    @LONGITUD INT,
    @CONTADOR INT,
    @RESULTADO INT;

-- Definir las palabras
SET @PALABRA1 = 'INFORMATICA';
SET @PALABRA2 = 'Informtaica';  -- Consideramos 'jseus' como ejemplo, podrías ajustar esta palabra según tus necesidades

SET @LONGITUD = LEN(@PALABRA1);
SET @CONTADOR = 1;
SET @RESULTADO = 0;

-- Imprimir las palabras
PRINT 'Palabra 1: ' + @PALABRA1;
PRINT 'Palabra 2: ' + @PALABRA2;
PRINT '';

-- Imprimir encabezado de la matriz
PRINT 'Matriz de Comparación:';
PRINT '----------------------';

-- Imprimir las letras y las comparaciones
WHILE @CONTADOR <= @LONGITUD
BEGIN
    PRINT SUBSTRING(@PALABRA1, @CONTADOR, 1) + '   ' + SUBSTRING(@PALABRA2, @CONTADOR, 1);
    SET @CONTADOR = @CONTADOR + 1;
END

-- Comparación de letras y contar coincidencias
SET @CONTADOR = 1;

WHILE @CONTADOR <= @LONGITUD
BEGIN
    IF SUBSTRING(@PALABRA1, @CONTADOR, 1) = SUBSTRING(@PALABRA2, @CONTADOR, 1)
    BEGIN
        SET @RESULTADO = @RESULTADO + 1;
    END

    SET @CONTADOR = @CONTADOR + 1;
END

-- Imprimir resultado
PRINT '';
PRINT 'Número de letras coincidentes entre ' + @PALABRA1 + ' y ' + @PALABRA2 + ': ' + CAST(@RESULTADO AS VARCHAR(10));
