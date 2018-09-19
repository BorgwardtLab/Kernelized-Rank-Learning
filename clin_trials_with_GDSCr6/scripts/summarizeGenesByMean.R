## Paul Geeleher
## All rights Reserved
## February 6, 2014

## This function will take a matrix with duplicate rownames()
## and summarize those rownames() by their mean.
summarizeGenesByMean <- function(exprMat)
{
  geneIds <- rownames(exprMat)
  t <- table(geneIds) # how many times is each gene name duplicated
  allNumDups <- unique(t)
  allNumDups <- allNumDups[-which(allNumDups == 1)]

  # create a *new* gene expression matrix with everything in the correct order....
  # start by just adding stuff that isn't duplicated
  exprMatUnique <- exprMat[which(geneIds %in% names(t[t == 1])), ]
  gnamesUnique <- geneIds[which(geneIds %in% names(t[t == 1]))]

  # add all the duplicated genes to the bottom of "exprMatUniqueHuman", summarizing as you go
  for(numDups in allNumDups) 
  {
    geneList <- names(which(t == numDups))
    
    for(i in 1:length(geneList))
    {
      exprMatUnique <- rbind(exprMatUnique, colMeans(exprMat[which(geneIds == geneList[i]), ]))
      gnamesUnique <- c(gnamesUnique, geneList[i])
      #print(i)
    }
  }

  rownames(exprMatUnique) <- gnamesUnique
  return(exprMatUnique)
}
