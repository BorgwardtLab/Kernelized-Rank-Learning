## Paul Geeleher
## All rights Reserved
## February 6, 2014

## a function to do variable selection on expression matrices.
## I.e. remove genes with low variation
## It returns a vector of row ids to keep. Note, rownames() must be specified.
doVariableSelection <- function(exprMat, removeLowVaryingGenes)
{
  vars <- apply(exprMat, 1, var)
  return(order(vars, decreasing=TRUE)[seq(1:as.integer(nrow(exprMat)*(1-removeLowVaryingGenes)))])
}










