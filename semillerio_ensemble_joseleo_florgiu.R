# Ensemble final mixfedeveroezemiguel + joseleoisunbaltrue con semillerio

require("data.table")
require("lightgbm")

PARAM <- list()
PARAM$experimento <- "ensemble_2_modelos_semillerio"

# Leer los tres archivos
archivo1 <- read.csv("~/buckets/b1/exp/ultimo_intento_semillerio_joseleounbaltrue_mode_corregido/ultimo_intento_semillerio_joseleounbaltrue_mode_corregido_predicciones_semillas.csv", header = TRUE)
archivo2 <- read.csv("~/buckets/b1/exp/intento-lightgbm-mix-1-SEMILLERIO_PLUS-promedios/intento-lightgbm-mix-1-SEMILLERIO_PLUS-promedios_predicciones_semillas.csv", header = TRUE)

if (!all(names(archivo1)[1:2] == names(archivo2)[1:2])) {
  stop("Las dos primeras columnas no son iguales en los tres archivos.")
}

# Calcular el promedio de las columnas 3 a 22
archivo1$promedio <- rowMeans(archivo1[, 3:22], na.rm = TRUE)
setorder(archivo1, numero_de_cliente)

# Calcular el promedio de las columnas 3 a 22
archivo2$promedio <- rowMeans(archivo2[, 3:22], na.rm = TRUE)
setorder(archivo2, numero_de_cliente)


# Crear un nuevo dataframe con las dos primeras columnas iguales y la tercera columna como promedio
predicciones <- data.frame(
  archivo2[, 1:2],
  Promedio = rowMeans(cbind(archivo1['promedio'], archivo2['promedio']), na.rm = TRUE)
)

# Guardar el nuevo dataframe en un nuevo archivo
write.table(predicciones, "nuevo_archivo.txt", sep = "\t", quote = FALSE, row.names = FALSE)

#------------------------------------------GENERO ENTREGA A KAGGLE------------------------------------------------#
# Ordeno por probabilidad descendente
setorder(predicciones, -Promedio)
setDT(predicciones)
setwd("~/buckets/b1/")
# Creo carpeta donde guardar los experimentos en caso de que no exista
dir.create("./exp/", showWarnings = FALSE)

# Creo carpeta donde guardar este experimento en caso de que no exista
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory de este experimento
setwd(paste0("./exp/", PARAM$experimento, "/"))


# Genero archivos variando la cantidad de estímulos
cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  predicciones[, Predicted := 0L]
  predicciones[1:envios, Predicted := 1L]
  
  fwrite(predicciones[, list(numero_de_cliente, Predicted)],
         file = paste0(PARAM$experimento, "_", envios, ".csv"),
         sep = ","
  )
}