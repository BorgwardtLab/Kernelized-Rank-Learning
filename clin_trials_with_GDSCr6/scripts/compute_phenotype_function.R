## Paul Geeleher
## All rights Reserved
## February 6, 2014


## Functions to predict a phenotype from microarray expression data of different platforms.


calcPhenotype <- function(testExprData, trainingExprData, trainingPtype, batchCorrect="eb", powerTransformPhenotype=TRUE, removeLowVaryingGenes=.2, minNumSamples=10, selection=-1, printOutput=TRUE)
{
  # Calculates a phenotype from gene expression microarray data, given a training set of expression data and known phenotype.
  #
  # Args:
  #   testExprData: The test data where the phenotype will be estimted. It is a matrix of expression levels, rows contain genes and columns contain samples, "rownames()" must be specified and must contain the same type of gene ids as "trainingExprData".
  #   trainingExprData: The training data. A matrix of expression levels, rows contain genes and columns contain samples, "rownames()" must be specified and must contain the same type of gene ids as "testExprData"
  #   trainingPtype: The known phenotype for "trainingExprData". A numeric vector which MUST be the same length as the number of columns of "trainingExprData".
  #   batchCorrect: How should training and test data matrices be homogenized. Choices are "eb" (default) for ComBat, "qn" for quantiles normalization or "none" for no homogenization.
  #   powerTransformPhenotype: Should the phenotype be power transformed before we fit the regression model? Default to TRUE, set to FALSE if the phenotype is already known to be highly normal.
  #   removeLowVaryingGenes: What proportion of low varying genes should be removed? 20% be default
  #   minNumSamples: How many training and test samples are requried. Print an error if below this threshold
  #   selection: How should duplicate gene ids be handled. Default is -1 which asks the user. 1 to summarize by their or 2 to disguard all duplicates.
  #
  # Returns:
  #   A vector of the estimated phenotype, in the same order as the columns of "testExprData".
  
  # check if the supplied data are of the correct classes
  if(class(testExprData) != "matrix") stop("ERROR: \"testExprData\" must be a matrix.");
  if(class(trainingExprData) != "matrix") stop("ERROR: \"trainingExprData\" must be a matrix.");
  if(class(trainingPtype) != "numeric") stop("ERROR: \"trainingPtype\" must be a numeric vector.");
  if(ncol(trainingExprData) != length(trainingPtype)) stop("The training phenotype must be of the same lenght as the number of columns of the training expressin matrix.");
  
  # check if an adequate number of training and test samples have been supplied.
  if((ncol(trainingExprData) < minNumSamples) || (ncol(testExprData) < minNumSamples))
  {
    stop(paste("There are less than", minNumSamples, "samples in your test or training set. It is strongly recommended that you use larger numbers of samples in order to (a) correct for batch effects and (b) fit a reliable model. To supress this message, change the \"minNumSamples\" parameter to this function."))
  }

  # Get the homogenized data
  homData <- homogenizeData(testExprData, trainingExprData, batchCorrect=batchCorrect, selection=selection, printOutput=printOutput)
  
  # Do variable selection if specified. By default we remove 20% of least varying genes.
  # Otherwise, keep all genes.
  if(removeLowVaryingGenes > 0 && removeLowVaryingGenes < 1)
  {
    keepRows <- doVariableSelection(cbind(homData$test, homData$train), removeLowVaryingGenes=removeLowVaryingGenes)
  }
  else 
    keepRows <- seq(1:nrow(homData$train))
  
  
  # PowerTranform phenotype if specified.
  offset = 0
  if(powerTransformPhenotype)
  {
    if(min(trainingPtype) < 0) # all numbers must be postive for a powerTranform to work, so make them positive.
    {
      offset <- -min(trainingPtype) + 1
      trainingPtype <- trainingPtype + offset
    }
    
    transForm <- powerTransform(trainingPtype)[[6]]
    trainingPtype <- trainingPtype^transForm
  }
  
  # create the Ridge Regression model on our training data
  if(printOutput) cat("\nFitting Ridge Regression model... ");
  trainFrame <- data.frame(Resp=trainingPtype, t(homData$train[keepRows, ]))
  rrModel <- linearRidge(Resp ~ ., data=trainFrame)
  if(printOutput) cat("Done\n\nCalculating predicted phenotype...");
  
  # calculate the relative contribution of each gene to the prediction
  # i might report these, I don't know if there's any point.
  totBeta <- sum(abs(coef(rrModel)))
  eachBeta <- abs(coef(rrModel))
  eachContribution <- eachBeta/totBeta
  
  # predict the new phenotype for the test data.
  # if there is a single test sample, there may be problems in predicting using the predict() function for the linearRidge package
  # This "if" statement provides a workaround
  if(class(homData$test) == "numeric")
  {
    n <- names(homData$test)
    homData$test <- matrix(homData$test, ncol=1)
    rownames(homData$test) <- n
    testFrame <- data.frame(t(homData$test[keepRows, ]))
    preds <- predict(rrModel, newdata=rbind(testFrame, testFrame))[1]
  }
  else #predict for more than one test sample
  {
    testFrame <- data.frame(t(homData$test[keepRows, ]))
    preds <- predict(rrModel, newdata=testFrame)
  }
  
  # if the response variable was transformed, untransform it now...
  if(powerTransformPhenotype)
  {
    preds <- preds^(1/transForm)
    preds <- preds - offset
  }
  if(printOutput) cat("Done\n\n");
  
  return(preds)
}


## This function uses X fold cross validation on the TrainingSet to estimate the accuracy of the phenotype prediction
## fold: How many fold cross-validation to use.
predictionAccuracyByCv <- function(testExprData, trainingExprData, trainingPtype, cvFold=-1, powerTransformPhenotype=TRUE, batchCorrect="eb", removeLowVaryingGenes=.2, minNumSamples=10, selection=1)
{

  # check if an adequate number of training and test samples have been supplied.
  if((ncol(trainingExprData) < minNumSamples) || (ncol(testExprData) < minNumSamples))
  {
    stop(paste("There are less than", minNumSamples, "samples in your test or training set. It is strongly recommended that you use larger numbers of samples in order to (a) correct for batch effects and (b) fit a reliable model. To supress this message, change the \"minNumSamples\" parameter to this function."))
  }
  
  # homogenize the data.
  homData <- homogenizeData(testExprData, trainingExprData, batchCorrect=batchCorrect, selection=selection) 
  
  nTrain <- ncol(trainingExprData)
  predPtype <- numeric() # a numeric vector to hold the predicted phenotypes for the CV subgroups
  
  # Perform either N fold cross validation or LOOCV, depending on the "cvFold" variable.
  if(cvFold == -1) # if we are doing leave-one-out cross-validation (LOOCV).
  {
    for(i in 1:nTrain)
    {
      
      testMatTemp <- matrix(homData$train[,i], ncol=1)
      rownames(testMatTemp) <- rownames(homData$train)
      #predPtype[i] <- calcPhenotype(testMatTemp, trainCvSet[,-i], trainingPtype[-i], batchCorrect="none", minNumSamples=0, selection=homData$selection, removeLowVaryingGenes=removeLowVaryingGenes, powerTransformPhenotype=powerTransformPhenotype)
      predPtype[i] <- calcPhenotype(testMatTemp, homData$train[,-i], trainingPtype[-i], batchCorrect="none", minNumSamples=0, selection=homData$selection, removeLowVaryingGenes=removeLowVaryingGenes, powerTransformPhenotype=powerTransformPhenotype, printOutput=FALSE)
      
      # print an update for each 20% of the this, this is slow, so we should give some updates....
      if(i %% as.integer(nTrain/5) == 0)
      cat(paste(i, "of" , nTrain, "iterations complete. \n"))
    }
  }
  else if(cvFold > 1) # if we are doing N-fold cross validation
  {
    randTestSamplesIndex <- sample(1:nTrain) # create a random order for samples
  
    # create a vector which indicates which samples are in which group... and split into list of groups
    sampleGroup <- rep(cvFold, nTrain)
    groupSize <- as.integer(nTrain / cvFold)
    for(j in 1:(cvFold-1)) { sampleGroup[(((j-1)*groupSize)+1):(j*groupSize)] <- rep(j, groupSize) }
    cvGroupIndexList <- split(randTestSamplesIndex, sampleGroup)
    
    # predict on each of the groups....
    for(j in 1:cvFold)
    {
      
      # create the ranomdly chosen "training" and "test" sets for cross validation
      testCvSet <- homData$train[, cvGroupIndexList[[j]]]
      trainCvSet <- homData$train[, unlist(cvGroupIndexList[-j])]
      trainPtypeCv <- trainingPtype[unlist(cvGroupIndexList[-j])]
      
      predPtype <- c(predPtype, calcPhenotype(testCvSet, trainCvSet, trainPtypeCv, batchCorrect="none", minNumSamples=0, selection=homData$selection, removeLowVaryingGenes=removeLowVaryingGenes, powerTransformPhenotype=powerTransformPhenotype))
      
      cat(paste("\n", j, " of ", cvFold, " iterations complete.", sep=""))
    }
    
    # re-order the predicted phenotypes correctly, as they were ranomized when this started.
    predPtype <- predPtype[order(randTestSamplesIndex)]
    
  }
  else {
    stop("Unrecognised value of \"cvFold\"")
  }
  
  finalData <- list(cvPtype=predPtype, realPtype=trainingPtype)
  class(finalData) <- "pRRopheticCv"
  
  return(finalData)
}