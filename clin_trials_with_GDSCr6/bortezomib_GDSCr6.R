### R code from vignette source 'bortezomib.Snw'

###################################################
### code chunk number 1: bortezomib.Snw:10-11
###################################################

# Set this if you want only the intersection of GDSCr2 and GDSCr6 cell lines
R2R6CCL=TRUE
# You can play with the preprocessing but the following is what Geeleher et al. 2014 used:
powTransP=TRUE
lowVarGeneThr=0.2
# Number of permutations
NUM_PERM=1

###################################################
### code chunk number 2: bortezomib.Snw:16-17
###################################################
#options(width=40)


###################################################
### code chunk number 3: bortezomib.Snw:20-26
###################################################
library("ridge")
library("sva")
library("car")
library("preprocessCore")
library("ROCR")
library("GEOquery")


###################################################
### code chunk number 4: bortezomib.Snw:31-36
###################################################
source("scripts/compute_phenotype_function.R")
source("scripts/summarizeGenesByMean.R")
source("scripts/homogenize_data.R")
source("scripts/do_variable_selection.R")


###################################################
### code chunk number 5: bortezomib.Snw:42-44
###################################################
load("data/bortGeo.RData") # loads the geo data "bortezomib_mas5"
# bortezomib_mas5 <- getGEO("GSE9782") # uncomment this line to download the data directly from GEO.


###################################################
### code chunk number 6: bortezomib.Snw:49-57
###################################################
exprDataU133a <- cbind(exprs(bortezomib_mas5[[1]]), exprs(bortezomib_mas5[[2]]))
bortIndex <- c(which(pData(phenoData(bortezomib_mas5[[1]]))[,"characteristics_ch1.1"] == "treatment = PS341"), 255 + which(pData(phenoData(bortezomib_mas5[[2]]))[,"characteristics_ch1.1"] == "treatment = PS341"))
dexIndex <- c(which(pData(phenoData(bortezomib_mas5[[1]]))[,"characteristics_ch1.1"] == "treatment = Dex"), 255 + which(pData(phenoData(bortezomib_mas5[[2]]))[,"characteristics_ch1.1"] == "treatment = Dex"))
studyIndex <- c(as.character(pData(bortezomib_mas5[[1]])[, "characteristics_ch1"]), as.character(pData(bortezomib_mas5[[2]])[, "characteristics_ch1"]))

# exprDataU133a <- exprs(bortezomib_mas5[[1]])
# bortIndex <- which(pData(phenoData(bortezomib_mas5[[1]]))[,"characteristics_ch1.1"] == "treatment = PS341")
# dexIndex <- which(pData(phenoData(bortezomib_mas5[[1]]))[,"characteristics_ch1.1"] == "treatment = Dex")


###################################################
### code chunk number 7: bortezomib.Snw:63-71
###################################################
library("hgu133a.db") # version 2.8.0
x <- hgu133aSYMBOL
mapped_probes <- mappedkeys(x)
names(mapped_probes) <- as.character(x[mapped_probes])
affy2sym <- as.character(x[mapped_probes])
sym2affy <- affy2sym
names(sym2affy) <- names(affy2sym)
rownames(exprDataU133a) <- sym2affy[rownames(exprDataU133a)]


###################################################
### code chunk number 8: bortezomib.Snw:76-77
###################################################
#X# load(file="../Data/GdscProcData/gdsc_brainarray_syms.RData")
#X# load(file="/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/gdsc_brainarray_syms.RData")
trainData <- read.csv("data/GEX.csv", as.is=TRUE, check.names=FALSE)
dim(trainData)


###################################################
### code chunk number 9: bortezomib.Snw:84-90
###################################################
#X# sensBortezomib <- read.csv("../Data/bortezomibData/sensitivity_data_for_drug_104.csv", as.is=TRUE)
#X# bortic50s <- sensBortezomib$"IC.50"
#X# names(bortic50s) <- sensBortezomib$"Cell.Line.Name"
#X# tissue <- sensBortezomib$"Tissue"
#X# names(tissue) <- sensBortezomib$"Cell.Line.Name"

#X# sensBortezomib <- read.csv("/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/geeleher_drugs/bortezomib.csv", as.is=TRUE)
#X# bortic50s <- sensBortezomib$"IC50"
#X# names(bortic50s) <- sensBortezomib$"Cell.line"
#X# tissue <- sensBortezomib$"Tissue"
#X# names(tissue) <- sensBortezomib$"Cell.line"

sensBortezomib <- read.csv("data/bortezomib_GEX_IC50.csv", as.is=TRUE)
bortic50s <- sensBortezomib$"IC50"
names(bortic50s) <- rownames(sensBortezomib)
length(bortic50s)

if (R2R6CCL) {
  r2r6 <- c('CESS', 'ETK-1', 'DSH1', 'ES1', 'IST-MEL1', 'IST-MES1', 'LOXIMVI', 'RS4-11', 'NCI-H1299', 'COLO-800', 'LS-411N', 'RKO', 'ES3', 'D-283MED', 'UACC-257', 'ES8', 'MPP-89', 'REH', 'RPMI-8866', 'NCI-H1355', 'Ramos-2G6-4C10', 'NCI-H1304', 'OCI-AML2', 'ONS-76', 'RL', 'PF-382', 'RXF393', 'OPM-2', 'QIMR-WIL', 'NCI-H1092', 'PSN1', 'RPMI-6666', 'P30-OHK', 'OCUB-M', 'RPMI-8226', 'NCI-H1155', 'OS-RC-2', 'KURAMOCHI', 'KINGS-1', 'LU-134-A', 'KY821', 'L-540', 'L-363', 'GI-1', 'MFH-ino', 'MEG-01', 'LAN-6', 'LXF-289', 'GI-ME-N', 'KS-1', 'LS-513', 'GR-ST', 'LB2241-RCC', 'LU-65', 'LAMA-84', 'KALS-1', 'KNS-81-FD', 'LU-139', 'LS-123', 'GCIY', 'GOTO', 'MONO-MAC-6', 'KGN', 'LC4-1', 'LS-1034', 'LOUCY', 'GT3TKB', 'MZ7-mel', 'TE-1', 'TE-8', 'MS-1', 'U-698-M', 'TE-10', 'TE-9', 'MOLT-4', 'TE-12', 'THP-1', 'ST486', 'ML-2', 'MHH-PREB-1', 'MRK-nu-1', 'SBC-1', 'TE-15', 'MMAC-SF', 'VA-ES-BJ', 'TE-5', 'TUR', 'SW872', 'U-87-MG', 'TE-6', 'CPC-N', 'BE-13', 'COLO-668', 'D-263MG', 'EM-2', 'AM-38', 'CAS-1', 'COR-L88', '697', 'COLO-320-HSR', 'D-502MG', 'A4-Fuk', 'ARH-77', 'BV-173', 'Calu-6', 'ATN-1', 'C2BBe1', 'CA46', 'DB', 'EW-22', 'A101D', 'COLO-824', 'CTV-1', 'DG-75', 'EW-11', 'BC-1', 'DJM-1', 'EW-1', 'SF539', 'SUP-T1', 'SK-LMS-1', 'HD-MY-Z', 'SW684', 'SHP-77', 'JVM-3', 'KNS-42', 'SW954', 'SK-UT-1', 'SW962', 'SR', 'JiyoyeP-2003', 'SCC-15', 'HAL-01', 'SKM-1', 'HUTU-80', 'HOP-62', 'IST-SL1', 'JAR', 'IM-9', 'IST-SL2', 'IMR-5', 'EVSA-T', 'CP66-MEL', 'EW-16', 'HL-60', 'ES7', 'DEL', 'EW-13', 'HCC2998', 'BB30-HNC', 'D-336MG', 'ECC12', 'EW-18', 'BB49-HNC', 'D-247MG', 'ES5', 'EB-3', 'BB65-RCC', 'CCRF-CEM', 'D-542MG', 'HCC1187', 'HCC2157', 'A253', 'COLO-684', 'DMS-79', 'EHEB', 'HCC2218', 'HH', 'HT-144', 'EC-GI-10', 'CGTH-W-1', 'EW-24', 'EB2', 'ALL-PO', 'DOHH-2', 'NCI-H510A', 'NCI-SNU-1', 'NB7', 'NMC-G1', 'NCI-SNU-16', 'SCH', 'NCI-H1838', 'NALM-6', 'NB14', 'no-11', 'NCI-H2227', 'NCI-H209', 'NCI-H747', 'NCI-SNU-5', 'NCI-H526', 'NKM-1', 'NB13', 'NB17', 'NCI-H82', 'NB12', 'NH-12', 'NB5', 'SCC-3', 'SF268', 'NCI-H23', 'NB69', 'no-10', 'CMK', 'NCI-H2141', 'SK-MEL-2', 'KARPAS-299', 'LP-1', 'NEC8', 'RPMI-8402', 'LB2518-MEL', 'COR-L279', 'LB373-MEL-D', 'NCI-H1436', 'UACC-812', 'RCC10RGB', 'Raji', 'TC-YIK', 'A3-KAW', 'KMOE-2', 'MHH-NB-11', 'NCI-H2196', 'SK-N-DZ', 'KASUMI-1', '8-MG-BA', 'CAL-148', 'P31-FUJ', 'SIMA', 'EW-12', 'KARPAS-45', 'ES6', 'NB1', 'LNCaP-Clone-FGC', 'LB996-RCC', 'NB6', 'JVM-2', 'NCI-H1770', 'NCI-H446', 'TK10', 'L-428', 'CW-2', 'GDM-1', 'MSTO-211H', 'TGBC1TKB', 'NOS-1', 'SJSA-1', 'MN-60', 'KM12', 'LB771-HNC', 'NCI-H345', 'NCI-H226', 'NCI-H716', 'LB831-BLC', 'HC-1')
  length(r2r6)
  bortic50s <- bortic50s[r2r6]
  names(bortic50s) <- r2r6
  length(bortic50s)
  if (any(is.na(bortic50s))){
    quit()
  }
}
###################################################
### code chunk number 10: bortezomib.Snw:96-104
###################################################
#X# pData <- read.delim("../Data/GdscPdata/E-MTAB-783.sdrf.txt", as.is=TRUE)
#X# pDataUnique <- pData[pData$Source.Name %in% names(which(table(pData$Source.Name) == 1)), ]
#X# rownames(pDataUnique) <- pDataUnique$Source.Name

#X# pData <- read.delim("/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/raw_expression_E-MTAB-3610/E-MTAB-3610.sdrf.txt", as.is=TRUE)
#X# pDataUnique <- pData[pData$Characteristics.cell.line. %in% names(which(table(pData$Characteristics.cell.line.) == 1)), ]
#X# rownames(pDataUnique) <- pDataUnique$Characteristics.cell.line.

commonCellLines <- names(trainData)[names(trainData) %in% names(bortic50s)]
length(commonCellLines)
if (R2R6CCL && length(commonCellLines) != length(r2r6)){
  quit()
}
#X# pDataUniqueOrd <- pDataUnique[commonCellLines, ]
bortic50sOrd <- bortic50s[commonCellLines]
#X# trainDataOrd <- gdsc_brainarray_syms[, pDataUniqueOrd$"Array.Data.File"]
trainDataOrd <- trainData[, commonCellLines]


###################################################
### code chunk number 11: bortezomib.Snw:110-111
###################################################
print(sum(grep("myeloma", sensBortezomib$Tissue), ignore.case=TRUE))

###################################################
### code chunk number 12: bortezomib.Snw:117-119
###################################################
trainDataOrdMat <- as.matrix(trainDataOrd)
rownames(trainDataOrdMat) <- rownames(trainDataOrd)
names(trainDataOrdMat) <- names(trainDataOrd)
predictedSensitivity133a <- calcPhenotype(exprDataU133a[, bortIndex], trainDataOrdMat, bortic50sOrd,
selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr)


###################################################
### code chunk number 13: bortezomib.Snw:125-129
###################################################
resp133a <- c(as.character(pData(bortezomib_mas5[[1]])[, "characteristics_ch1.8"]),
as.character(pData(bortezomib_mas5[[2]])[, "characteristics_ch1.8"]))[bortIndex]
t.test(predictedSensitivity133a[resp133a == "PGx_Responder = NR"],
predictedSensitivity133a[resp133a == "PGx_Responder = R"], alternative="greater")



###################################################
### code chunk number 14: fig4aPlot
###################################################
lTwoa <- list("Responder"=predictedSensitivity133a[resp133a == "PGx_Responder = R"],
"Non-responder"=predictedSensitivity133a[resp133a == "PGx_Responder = NR"])
boxplot(lTwoa, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(a)")
stripchart(lTwoa, vertical=TRUE, pch=20, method="jitter", add=TRUE)

###################################################
### code chunk number 15: fig3a
###################################################
lTwoa <- list("Responder"=predictedSensitivity133a[resp133a == "PGx_Responder = R"],
"Non-responder"=predictedSensitivity133a[resp133a == "PGx_Responder = NR"])
boxplot(lTwoa, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(a)")
stripchart(lTwoa, vertical=TRUE, pch=20, method="jitter", add=TRUE)


###################################################
### code chunk number 16: bortezomib.Snw:156-160
###################################################
predA <- prediction(c(lTwoa[[1]], lTwoa[[2]]), c(rep("sens", length(lTwoa[[1]])), 
rep("res", length(lTwoa[[2]]))), label.ordering=c("sens", "res"))
perfA <- performance(predA, measure = "tpr", x.measure = "fpr")
print(paste("AUC:", performance(predA, measure = "auc")@"y.values"[[1]]))


###################################################
### code chunk number 17: bortezomib.Snw:164-175
###################################################
aucs <- numeric()
for(i in 1:NUM_PERM)
{
  predPerm <- prediction(c(lTwoa[[1]], lTwoa[[2]]), 
  sample(c(rep("sens", length(lTwoa[[1]])), 
  rep("res", length(lTwoa[[2]])))), label.ordering=c("sens", "res"))
  aucs[i] <- performance(predPerm, measure = "auc")@"y.values"[[1]]
}
permutationP <- sum(aucs > performance(predA, 
measure = "auc")@"y.values"[[1]])/NUM_PERM
print(paste("ROC Permuatation P-value:", permutationP))


###################################################
### code chunk number 18: fig4bPlot
###################################################
plot(perfA, main="(b)")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 19: fig4b
###################################################
plot(perfA, main="(b)")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 20: fig4cPlot
###################################################
fullResp133a <- c(as.character(pData(bortezomib_mas5[[1]])$"characteristics_ch1.7"),
as.character(pData(bortezomib_mas5[[2]])$"characteristics_ch1.7"))[bortIndex]
l <- split(predictedSensitivity133a, fullResp133a)
lOrd <- l[c("PGx_Response = CR", "PGx_Response = PR", "PGx_Response = MR",
"PGx_Response = NC", "PGx_Response = PD")]
names(lOrd) <- c("CR", "PR", "MR", "NC", "PD")
boxplot(lOrd, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(c)")
stripchart(lOrd, vertical=TRUE, pch=20, method="jitter", add=TRUE)


###################################################
### code chunk number 21: fig4c
###################################################
fullResp133a <- c(as.character(pData(bortezomib_mas5[[1]])$"characteristics_ch1.7"),
as.character(pData(bortezomib_mas5[[2]])$"characteristics_ch1.7"))[bortIndex]
l <- split(predictedSensitivity133a, fullResp133a)
lOrd <- l[c("PGx_Response = CR", "PGx_Response = PR", "PGx_Response = MR",
"PGx_Response = NC", "PGx_Response = PD")]
names(lOrd) <- c("CR", "PR", "MR", "NC", "PD")
boxplot(lOrd, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(c)")
stripchart(lOrd, vertical=TRUE, pch=20, method="jitter", add=TRUE)


###################################################
### code chunk number 22: bortezomib.Snw:227-231
###################################################
clinResp133a <- c(rep(1, length(lOrd$"CR")), rep(2, length(lOrd$"PR")), 
rep(3,length(lOrd$"MR") ), rep(4, length(lOrd$"NC")), rep(5, length(lOrd$"PD")))
predSens133a <- do.call(c, lOrd)
summary(lm(clinResp133a~predSens133a))


###################################################
### code chunk number 23: bortezomib.Snw:236-243
###################################################
predictedSensitivity133a_DEX <- calcPhenotype(exprDataU133a[, dexIndex], trainDataOrdMat, bortic50sOrd,
selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr)

resp133a_DEX <- c(as.character(pData(bortezomib_mas5[[1]])[, "characteristics_ch1.8"]),
as.character(pData(bortezomib_mas5[[2]])[, "characteristics_ch1.8"]))[dexIndex]
t.test(predictedSensitivity133a_DEX[resp133a_DEX == "PGx_Responder = NR"],
predictedSensitivity133a_DEX[resp133a_DEX == "PGx_Responder = R"])


###################################################
### code chunk number 24: bortezomib.Snw:248-254
###################################################
bortExprMat_a <- exprDataU133a[, bortIndex]
bortStudyInd <- studyIndex[bortIndex]
pred39_a <- calcPhenotype(bortExprMat_a[, bortStudyInd == "studyCode = 39"], trainDataOrdMat, bortic50sOrd, selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr) # this works best
t.test(pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = NR"], 
pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = R"], alternative="greater")


###################################################
### code chunk number 25: bortezomib.Snw:258-278
###################################################
# get the response data and encode as a numeric vector.
resps <- resp133a[ bortStudyInd == "studyCode = 39"]
resps[resps == "PGx_Responder = NR"] <- 0
resps[resps == "PGx_Responder = R"] <- 1
resps[resps == "PGx_Responder = IE"] <- 2
respsNum <- as.numeric(resps)

thresholds <- sort(pred39_a) # the various cut-points that we will test
classificationAccuracy_a <- numeric()
numCorrectlyClassified <- numeric()
for(i in 1:length(pred39_a))
{
  isSens <- as.numeric(pred39_a <= thresholds[i]) # create a vector indicating whether each sample is classified as "sensitive" or resistant at this threshold.
  numCorrectlyClassified[i] <- sum(isSens == respsNum)
  classificationAccuracy_a[i] <- numCorrectlyClassified[i] / (sum(respsNum == 0)+sum(respsNum == 1)) # compute the proportion of these the above that are correctly classified.
}
maxAccuracy <- max(classificationAccuracy_a)
optimalCutpoint <- thresholds[classificationAccuracy_a == maxAccuracy]
print(maxAccuracy)
print(optimalCutpoint)


###################################################
### code chunk number 26: bortezomib.Snw:282-287
###################################################
unbiasedCutpoint <- mean(bortic50sOrd)
isSens <- as.numeric(pred39_a <= unbiasedCutpoint) # create a vector indicating whether each sample is classified as "sensitive" or resistant at this threshold.
numCorrectlyClassified <- sum(isSens == respsNum)
classificationAccuracy_a_unbiased <- numCorrectlyClassified / (sum(respsNum == 0)+sum(respsNum == 1)) # compute the proportion of these the above that are correctly classified.
print(classificationAccuracy_a_unbiased)


###################################################
### code chunk number 27: fig4dPlot
###################################################
lTwoa_039 <- list("Responder"=pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = R"],
"Non-responder"=pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = NR"])
boxplot(lTwoa_039, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(d)")
stripchart(lTwoa_039, vertical=TRUE, pch=20, method="jitter", add=TRUE)
abline(h=optimalCutpoint, lwd=2)
abline(h=-5.565168, lty=2)
abline(h=-4.532760, lty=2)
abline(h=unbiasedCutpoint, lty=2, col="red")


###################################################
### code chunk number 28: bortezomib.Snw:306-308
###################################################
#loocvOut_bort <- predictionAccuracyByCv(exprDataU133a[, bortIndex], trainDataOrdMat, bortic50sOrd)
#cor.test(loocvOut_bort$cvPtype, loocvOut_bort$realPtype)


###################################################
### code chunk number 29: bortezomib.Snw:315-327
###################################################
exprDataU133b <- cbind(exprs(bortezomib_mas5[[3]]), exprs(bortezomib_mas5[[4]]))
library("hgu133b.db")
x <- hgu133bSYMBOL
mapped_probes <- mappedkeys(x) # Get the probe identifiers that are mapped to a gene 
names(mapped_probes) <- as.character(x[mapped_probes])
affy2sym <- as.character(x[mapped_probes])
sym2affy <- affy2sym
names(sym2affy) <- names(affy2sym)
rownames(exprDataU133b) <- sym2affy[rownames(exprDataU133b)]

predictedSensitivityU133b <- calcPhenotype(exprDataU133b[, bortIndex], trainDataOrdMat, bortic50sOrd, 
selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr)


###################################################
### code chunk number 30: bortezomib.Snw:332-336
###################################################
respU133b <- c(as.character(pData(bortezomib_mas5[[3]])[, "characteristics_ch1.8"]),
as.character(pData(bortezomib_mas5[[4]])[, "characteristics_ch1.8"]))[bortIndex]
t.test(predictedSensitivityU133b[respU133b == "PGx_Responder = NR"], 
predictedSensitivityU133b[respU133b == "PGx_Responder = R"], alternative="greater")


###################################################
### code chunk number 31: fig4ePlot
###################################################
l_U133b <- list("Responder"=predictedSensitivityU133b[respU133b == "PGx_Responder = R"]
, "Non-responder"=predictedSensitivityU133b[respU133b == "PGx_Responder = NR"])
boxplot(l_U133b, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="", ylim=c(-7, -2.5))
stripchart(l_U133b, vertical=TRUE, pch=20, method="jitter", add=TRUE)


###################################################
### code chunk number 32: fig4e
###################################################
l_U133b <- list("Responder"=predictedSensitivityU133b[respU133b == "PGx_Responder = R"]
, "Non-responder"=predictedSensitivityU133b[respU133b == "PGx_Responder = NR"])
boxplot(l_U133b, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="", ylim=c(-7, -2.5))
stripchart(l_U133b, vertical=TRUE, pch=20, method="jitter", add=TRUE)

###################################################
### code chunk number 33: bortezomib.Snw:363-373
###################################################
fullResp133b <- c(as.character(pData(bortezomib_mas5[[3]])$"characteristics_ch1.7"), 
as.character(pData(bortezomib_mas5[[4]])$"characteristics_ch1.7"))[bortIndex]
lb <- split(predictedSensitivityU133b, fullResp133b)
lOrdb <- lb[c("PGx_Response = CR", "PGx_Response = PR", "PGx_Response = MR", 
"PGx_Response = NC", "PGx_Response = PD")]
names(lOrdb) <- c("CR", "PR", "MR", "NC", "PD")
clinResp133b <- c(rep(1, length(lOrdb$"CR")), rep(2, length(lOrdb$"PR")), 
rep(3,length(lOrdb$"MR") ), rep(4, length(lOrdb$"NC")), rep(5, length(lOrdb$"PD")))
predSens133b <- do.call(c, lOrdb)
summary(lm(clinResp133b~predSens133b))

###################################################
### code chunk number 34: fig4fPlot
###################################################
boxplot(lOrdb, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)", 
main="", ylim=c(-7, -2.5))
stripchart(lOrdb, vertical=TRUE, pch=20, method="jitter", add=TRUE)



###################################################
### code chunk number 35: fig4f
###################################################
boxplot(lOrdb, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)", 
main="", ylim=c(-7, -2.5))
stripchart(lOrdb, vertical=TRUE, pch=20, method="jitter", add=TRUE)


###################################################
### code chunk number 36: bortezomib.Snw:398-403
###################################################
predB <- prediction(c(l_U133b[[1]], l_U133b[[2]]), c(rep("sens", length(l_U133b[[1]])), 
rep("res", length(l_U133b[[2]]))), 
label.ordering=c("sens", "res"))
perfB <- performance(predB, measure = "tpr", x.measure = "fpr")
print(paste("AUC:", performance(predB, measure = "auc")@"y.values"[[1]]))


###################################################
### code chunk number 37: bortezomib.Snw:407-418
###################################################
aucs <- numeric()
for(i in 1:NUM_PERM)
{
  predPerm <- prediction(c(l_U133b[[1]], l_U133b[[2]]), 
  sample(c(rep("sens", length(l_U133b[[1]])), 
  rep("res", length(l_U133b[[2]])))), label.ordering=c("sens", "res"))
  aucs[i] <- performance(predPerm, measure = "auc")@"y.values"[[1]]
}
permutationP <- sum(aucs > performance(predB, 
measure = "auc")@"y.values"[[1]])/NUM_PERM
print(paste("ROC Permuatation P-value:", permutationP))


###################################################
### code chunk number 38: fig4gPlot
###################################################
plot(perfB, main="")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 39: fig4g
###################################################
plot(perfB, main="")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 40: bortezomib.Snw:446-453
###################################################
predictedSensitivity133b_DEX <- calcPhenotype(exprDataU133b[, dexIndex], trainDataOrdMat, bortic50sOrd,
selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr)

resp133b_DEX <- c(as.character(pData(bortezomib_mas5[[3]])[, "characteristics_ch1.8"]),
as.character(pData(bortezomib_mas5[[4]])[, "characteristics_ch1.8"]))[dexIndex]
t.test(predictedSensitivity133b_DEX[resp133b_DEX == "PGx_Responder = NR"],
predictedSensitivity133b_DEX[resp133b_DEX == "PGx_Responder = R"])


###################################################
### code chunk number 41: bortezomib.Snw:458-464
###################################################
bortExprMat <- exprDataU133b[, bortIndex]
bortStudyInd <- studyIndex[bortIndex]
pred39_b <- calcPhenotype(bortExprMat[, bortStudyInd == "studyCode = 39"], trainDataOrdMat, bortic50sOrd, selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr) # this works best
t.test(pred39_b[respU133b[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = NR"], 
pred39_b[respU133b[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = R"], alternative="greater")


###################################################
### code chunk number 42: bortezomib.Snw:468-484
###################################################
# get the response data and encode as a numeric vector.
resps <- respU133b[ bortStudyInd == "studyCode = 39"]
resps[resps == "PGx_Responder = NR"] <- 0
resps[resps == "PGx_Responder = R"] <- 1
resps[resps == "PGx_Responder = IE"] <- 2
respsNum <- as.numeric(resps)

thresholds <- sort(pred39_b) # the various cut-points that we will test
classificationAccuracy <- numeric()
for(i in 1:length(pred39_b))
{
  isSens <- as.numeric(pred39_b <= thresholds[i]) # create a vector indicating whether each sample is classified as "sensitive" or resistant at this threshold.
  classificationAccuracy[i] <- sum(isSens == respsNum) / (sum(respsNum == 0)+sum(respsNum == 1)) # compute the proportion of these the above that are correctly classified.
}

print(max(classificationAccuracy))


###################################################
### code chunk number 43: bortezomib.Snw:502-509
###################################################
pdf(file="fig4.pdf", width=9, height=9)
par(mfrow=c(2,2))
lTwoa <- list("Responder"=predictedSensitivity133a[resp133a == "PGx_Responder = R"],
"Non-responder"=predictedSensitivity133a[resp133a == "PGx_Responder = NR"])
boxplot(lTwoa, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(a)")
stripchart(lTwoa, vertical=TRUE, pch=20, method="jitter", add=TRUE)
plot(perfA, main="(b)")
abline(0, 1, col="grey", lty=2)
fullResp133a <- c(as.character(pData(bortezomib_mas5[[1]])$"characteristics_ch1.7"),
as.character(pData(bortezomib_mas5[[2]])$"characteristics_ch1.7"))[bortIndex]
l <- split(predictedSensitivity133a, fullResp133a)
lOrd <- l[c("PGx_Response = CR", "PGx_Response = PR", "PGx_Response = MR",
"PGx_Response = NC", "PGx_Response = PD")]
names(lOrd) <- c("CR", "PR", "MR", "NC", "PD")
boxplot(lOrd, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(c)")
stripchart(lOrd, vertical=TRUE, pch=20, method="jitter", add=TRUE)
lTwoa_039 <- list("Responder"=pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = R"],
"Non-responder"=pred39_a[resp133a[ bortStudyInd == "studyCode = 39"] == "PGx_Responder = NR"])
boxplot(lTwoa_039, outline=FALSE, border="grey", ylab="Predicted Sensitivity (log(IC50)",
main="(d)")
stripchart(lTwoa_039, vertical=TRUE, pch=20, method="jitter", add=TRUE)
abline(h=optimalCutpoint, lwd=2)
abline(h=-5.565168, lty=2)
abline(h=-4.532760, lty=2)
abline(h=unbiasedCutpoint, lty=2, col="red")
dev.off()


###################################################
### code chunk number 44: bortezomib.Snw:515-516
###################################################
sessionInfo()


