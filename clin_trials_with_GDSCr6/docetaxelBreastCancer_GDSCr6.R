### R code from vignette source 'docetaxelBreastCancer.Snw'

###################################################
### code chunk number 1: docetaxelBreastCancer.Snw:9-10

# Set this if you want only the intersection of GDSCr2 and GDSCr6 cell lines
R2R6CCL=TRUE
# You can play with the preprocessing but the following is what Geeleher et al. 2014 used:
powTransP=TRUE
lowVarGeneThr=0.2

###################################################
### code chunk number 2: docetaxelBreastCancer.Snw:15-20
###################################################
library("ridge")
library("sva")
library("car")
library("preprocessCore")
library("ROCR")


###################################################
### code chunk number 3: docetaxelBreastCancer.Snw:25-29
###################################################
source("scripts/compute_phenotype_function.R")
source("scripts/summarizeGenesByMean.R")
source("scripts/homogenize_data.R")
source("scripts/do_variable_selection.R")


###################################################
### code chunk number 4: docetaxelBreastCancer.Snw:34-36
###################################################
#load(file="/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/gdsc_brainarray_syms.RData")
trainData <- read.csv("data/GEX.csv", as.is=TRUE, check.names=FALSE)
dim(trainData)
load(file="data/doce_rma_syms_brainArray.RData")


###################################################
### code chunk number 5: docetaxelBreastCancer.Snw:42-46
###################################################
#sensDoce <- read.csv("/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/geeleher_drugs/docetaxel.csv", as.is=TRUE)
sensDoce <- read.csv("data/docetaxel_GEX_IC50.csv", as.is=TRUE)
doceic50s <- sensDoce$"IC50"
names(doceic50s) <- rownames(sensDoce)
length(doceic50s)

if (R2R6CCL) {
  r2r6 = c('CAL-120', 'CAL-51', 'A2058', 'HCC1419', 'ETK-1', 'SW1783', 'DSH1', 'ES1', 'IST-MEL1', '5637', 'HMV-II', 'M14', 'HT-29', 'NCI-H441', 'HCT-116', 'KYSE-450', 'NCI-H2052', 'HOS', 'NCI-H28', 'NCI-H358', 'NCI-H727', 'COLO-800', 'Saos-2', 'LS-411N', 'RKO', 'WM-115', 'D-283MED', '639-V', 'OE19', 'NCI-H2452', 'UACC-62', 'COLO-680N', 'CAL-33', 'UACC-257', 'PA-1', 'NCI-H661', 'ES8', 'SW1417', 'MPP-89', 'RPMI-8866', 'NCI-H1792', 'NCI-H1048', 'OAW-42', 'OE33', 'OVCAR-8', 'Ramos-2G6-4C10', 'RH-18', 'NCI-H1304', 'OCI-AML2', 'ONS-76', 'NCI-H1755', 'RXF393', 'RMG-I', 'RVH-421', 'PFSK-1', 'RCM-1', 'RERF-LC-MS', 'RPMI-2650', 'QIMR-WIL', 'NCI-H1092', 'NCI-H1651', 'PSN1', 'RO82-W-1', 'P12-ICHIKAWA', 'NCI-H1563', 'P30-OHK', 'OCUB-M', 'RPMI-8226', 'PANC-03-27', 'OC-314', 'NCI-H1993', 'NCI-H1155', 'OS-RC-2', 'KYSE-520', 'KURAMOCHI', 'KINGS-1', 'LCLC-97TM1', 'LU-134-A', 'GMS-10', 'KY821', 'LCLC-103H', 'LN-405', 'GAK', 'KYSE-410', 'L-363', 'FADU', 'G-401', 'GI-1', 'MFH-ino', 'MEG-01', 'KYSE-510', 'LAN-6', 'LXF-289', 'GI-ME-N', 'MFE-280', 'KS-1', 'LU-135', 'LS-513', 'G-361', 'GR-ST', 'MOLT-16', 'Mewo', 'KYSE-70', 'LB2241-RCC', 'LU-65', 'LAMA-84', 'MOLT-13', 'KYSE-140', 'KNS-81-FD', 'LU-139', 'LS-123', 'GCIY', 'GOTO', 'MEL-HO', 'KGN', 'LS-1034', 'GT3TKB', 'U-118-MG', 'YH-13', 'TE-1', 'TE-8', 'SiHa', 'SCC-9', 'MS-1', 'YAPC', 'TE-9', 'SNU-449', 'SW780', 'MOLT-4', 'TGBC11TKB', 'SK-HEP-1', 'MG-63', 'YKG-1', 'TE-12', 'SK-OV-3', 'ML-2', 'MHH-PREB-1', 'TE-11', 'SK-MEL-24', 'MCF7', 'UM-UC-3', 'SBC-1', 'SW1990', 'SW620', 'T47D', 'MKN28', 'MMAC-SF', 'U251', 'VA-ES-BJ', 'SAS', 'TE-5', 'SNU-387', 'T98G', 'U-87-MG', 'UMC-11', 'TE-6', 'T84', 'BT-20', 'Ca9-22', 'CHP-212', 'DK-MG', 'EGI-1', '23132-87', 'ACHN', 'BE-13', 'BHT-101', 'CAKI-1', 'COLO-679', 'COLO-668', 'D-263MG', 'EM-2', 'AM-38', 'CAS-1', 'COLO-678', 'Capan-2', 'COR-L88', 'DOK', '697', 'AN3-CA', 'COLO-320-HSR', 'Ca-Ski', 'D-502MG', 'DU-145', 'A4-Fuk', 'BV-173', 'Calu-6', 'COR-L23', 'Daoy', 'DoTc2-4510', 'A431', 'ATN-1', 'C2BBe1', 'CAPAN-1', 'CFPAC-1', 'Detroit562', 'DB', 'EW-22', 'A101D', 'A549', 'BALL-1', 'CHL-1', 'CAL-39', 'COLO-824', 'CTV-1', 'EFM-19', 'EW-11', 'DJM-1', 'EFO-27', 'EW-1', 'MEL-JUSO', 'SF539', 'HD-MY-Z', 'SW684', 'HOP-92', 'JEG-3', 'MDA-MB-231', 'SHP-77', 'SW948', 'JVM-3', 'MDA-MB-361', 'KNS-42', 'HSC-4', 'HN', 'HT-1080', 'SW954', 'MDA-MB-453', 'SK-UT-1', 'HT-1376', 'SW962', 'IA-LM', 'SBC-5', 'SKG-IIIa', 'KNS-62', 'H4', 'HT55', 'HuP-T4', 'HAL-01', 'SW1573', 'SK-MEL-30', 'HOP-62', 'IST-SL1', 'IGROV-1', 'CaR-1', 'CP66-MEL', 'EW-16', 'HL-60', 'HCC1937', 'ES7', 'A2780', 'DEL', 'EW-13', 'HuO9', 'HCC2998', 'EFO-21', 'BB30-HNC', 'D-336MG', 'HuP-T3', 'HEC-1', 'HGC-27', 'H9', 'EW-18', 'A427', 'BB49-HNC', 'D-247MG', 'ES5', 'A498', 'BB65-RCC', 'CCRF-CEM', 'D-542MG', 'HuCCT1', 'HCC1954', 'EPLC-272H', 'COLO-684', 'DMS-79', 'HCC2218', 'HH', 'EC-GI-10', 'ABC-1', 'CAL-27', 'CGTH-W-1', 'EW-24', 'HCC70', 'ALL-PO', 'DOHH-2', 'HuH-7', 'HT-3', 'NCI-H2347', 'NCI-H510A', 'NCI-H810', 'NCI-SNU-1', 'NMC-G1', 'NCI-H1666', 'SW1710', 'NCI-H2405', 'NCI-H2029', 'SCH', 'NCI-H1838', 'NCI-H650', 'NB14', 'no-11', 'SNU-423', 'NCI-H1703', 'NCI-H209', 'NCI-H2291', 'NCI-H747', 'NCI-SNU-5', 'NY', 'SCC-25', 'NCI-H1693', 'NCI-H520', 'NCI-H526', 'NKM-1', 'NB13', 'SNU-C2B', 'NB17', 'SW13', 'NCI-H2342', 'NCI-H630', 'NCI-H596', 'NCI-H82', 'NB12', 'NH-12', 'NB5', 'SW48', 'SF268', 'NCI-H23', 'NB69', 'no-10', 'SCC-4', 'SW982', 'SK-MEL-2', 'Hs-578-T', 'KARPAS-299', 'LK-2', 'NEC8', 'LB2518-MEL', 'TYK-nu', 'LB373-MEL-D', 'MDA-MB-415', 'FTC-133', 'RCC10RGB', 'MKN45', 'SF295', 'AsPC-1', 'NCI-N87', 'MIA-PaCa-2', 'A3-KAW', 'KMOE-2', 'MHH-NB-11', 'BxPC-3', 'HuO-3N1', 'Calu-3', 'SK-N-DZ', 'KASUMI-1', 'SK-N-AS', '8-MG-BA', 'CAL-12T', 'LoVo', 'KARPAS-45', 'ES6', 'NCI-H292', 'ESS-1', 'HCC1395', 'NB6', 'JVM-2', 'DBTRG-05MG', 'KP-N-YN', 'NCI-H1770', 'NCI-H446', 'TK10', 'L-428', 'CW-2', 'HCC38', 'D-423MG', '8505C', 'M059J', 'CAL-62', 'SW626', 'DMS-273', 'MSTO-211H', 'TGBC1TKB', 'NOS-1', 'SJSA-1', 'MN-60', 'KM12', '22RV1', 'LB771-HNC', 'VMRC-RCZ', 'KU-19-19', 'U-2-OS', 'D-566MG', 'NCI-H226', 'LB831-BLC', 'HC-1')
  length(r2r6)
  doceic50s <- doceic50s[r2r6]
  names(doceic50s) <- r2r6
  length(doceic50s)
  if (any(is.na(doceic50s))){
    quit()
  }
}

###################################################
### code chunk number 6: docetaxelBreastCancer.Snw:50-58
###################################################
#pData <- read.delim("/links/groups/borgwardt/Projects/drug_sensitivity/final/data/GDSC/raw_expression_E-MTAB-3610/E-MTAB-3610.sdrf.txt", as.is=TRUE)
#pDataUnique <- pData[pData$Characteristics.cell.line. %in% names(which(table(pData$Characteristics.cell.line.) 
#== 1)), ]
#rownames(pDataUnique) <- pDataUnique$Characteristics.cell.line.
commonCellLines <- names(trainData)[names(trainData) %in% names(doceic50s)]
length(commonCellLines)
if (R2R6CCL && length(commonCellLines) != length(r2r6)){
  quit()
}
#pDataUniqueOrd <- pDataUnique[commonCellLines, ]
doceic50sOrd <- doceic50s[commonCellLines]
#trainDataOrd <- gdsc_brainarray_syms[, pDataUniqueOrd$"Array.Data.File"]
trainDataOrd <- trainData[, commonCellLines]

###################################################
### code chunk number 7: docetaxelBreastCancer.Snw:64-66
###################################################
mat <- as.matrix(trainDataOrd)
rownames(mat) <- rownames(trainDataOrd)
names(mat) <- names(trainDataOrd)
predictedSensitivity <- calcPhenotype(doceVivoNorm_syms, mat, doceic50sOrd, selection=1, powerTransformPhenotype=powTransP, removeLowVaryingGenes=lowVarGeneThr)


###################################################
### code chunk number 8: docetaxelBreastCancer.Snw:70-71
###################################################
t.test(predictedSensitivity[1:10], predictedSensitivity[11:24], alternative="less")


###################################################
### code chunk number 9: fig3aPlot
###################################################
stripchart(list("Sensitive in vivo"=predictedSensitivity[1:10], 
"Resistant in vivo"=predictedSensitivity[11:24]), vertical=TRUE, 
method="jitter", pch=20, ylab="Predicted Sensitivity (log(IC50))", main="(a)")


###################################################
### code chunk number 10: fig3a
###################################################
stripchart(list("Sensitive in vivo"=predictedSensitivity[1:10], 
"Resistant in vivo"=predictedSensitivity[11:24]), vertical=TRUE, 
method="jitter", pch=20, ylab="Predicted Sensitivity (log(IC50))", main="(a)")


###################################################
### code chunk number 11: docetaxelBreastCancer.Snw:96-100
###################################################
pred <- prediction(predictedSensitivity, c(rep("sens", 10), rep("res", 14)), 
label.ordering=c("sens", "res"))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
print(paste("AUC:", performance(pred, measure = "auc")@"y.values"[[1]]))


###################################################
### code chunk number 12: fig3bPlot
###################################################
plot(perf, main="(b)")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 13: fig3b
###################################################
plot(perf, main="(b)")
abline(0, 1, col="grey", lty=2)


###################################################
### code chunk number 14: docetaxelBreastCancer.Snw:124-126
###################################################
#loocvOut_doce <- predictionAccuracyByCv(doceVivoNorm_syms, trainDataOrd, doceic50sOrd)
#cor.test(loocvOut_doce$cvPtype, loocvOut_doce$realPtype)


###################################################
### code chunk number 15: docetaxelBreastCancer.Snw:130-135
###################################################
pdf(width=9, height=4.5, file="Fig3.pdf")
par(mfrow=c(1,2))
stripchart(list("Sensitive in vivo"=predictedSensitivity[1:10], 
"Resistant in vivo"=predictedSensitivity[11:24]), vertical=TRUE, 
method="jitter", pch=20, ylab="Predicted Sensitivity (log(IC50))", main="(a)")
plot(perf, main="(b)")
abline(0, 1, col="grey", lty=2)
dev.off()


###################################################
### code chunk number 16: docetaxelBreastCancer.Snw:141-142
###################################################
sessionInfo()


