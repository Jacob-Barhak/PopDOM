# -*- coding: utf-8 -*-
#############################################################################
#Copyright (C) 2018-2019 Jacob Barhak, Aaron Garrett
# 
#This file is part of the Population Disease Occurrence Models . The Population Disease Occurrence Models is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#The Population Disease Occurrence Models is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#See the GNU General Public License for more details.
#############################################################################
"""
This script creates the computations for the MODSIM 2019 paper 
Population Disease Occurrence Models Using Evolutionary Computation
by: Olaf Dammann, Anselm Blumer, Jacob Barhak, Aaron Garrett

Authors Contact information:

Aaron Garrett
aaron.lee.garrett@gmail.com
http://sites.wofford.edu/garrettal/

Jacob Barhak Ph.D.
jacob.barhak@gmail.com
http://sites.google.com/site/jacobbarhak/
"""


from __future__ import division
import inspyred
import numpy
from sklearn.linear_model import LogisticRegression
import copy
import pandas
import dask
import dask.distributed
import yaml
import sys
import os
import operator
import bokeh
import bokeh.plotting # import figure
import bokeh.resources # import CDN
import bokeh.embed # import file_html
import bokeh.layouts #import column , row
import bokeh.models #import Range1d , HoverTool, ColumnDataSource
import bokeh.io # import export_png
import webbrowser



def ApplyFunctionToNonNoneElements (InputList, FunctionToApply):
    "Applies function to non None elements"
    OutputList = []
    for Entry in InputList:
        if Entry == None:
            Value = None
        else:
            Value = FunctionToApply(Entry)
        OutputList.append(Value)
    return OutputList



def OddsRatiosByDefinition(Candidate, CalculateDivisionRatiosInstead = False):
    " Calculate Odds ratio by using definition "
    # perform regression for any combination of risk factors/outcome
    # organize results in descending order such that 
    # Rn/Rn-1 is first and R2/R1 is last
    # if CalculateDivisionRatiosInstead is set then the output
    # would be ratios rather than odds ratios 
    Factors = len(Candidate)
    CalculatedOddsRatios = []
    for OutcomeIndex in reversed(range(Factors)):
        for ValuesIndex in reversed(range(OutcomeIndex)):
            Values = Candidate[ValuesIndex]
            Outcomes = Candidate[OutcomeIndex]
            NotValues = numpy.logical_not(Values)
            NotOutcomes = numpy.logical_not(Outcomes)
            A = sum(numpy.logical_and(Values,Outcomes))
            B = sum(numpy.logical_and(Values,NotOutcomes))
            C = sum(numpy.logical_and(NotValues,Outcomes))
            D = sum(numpy.logical_and(NotValues,NotOutcomes))
            if CalculateDivisionRatiosInstead:
                CalculatedOddsRatiosComponent = A/B
            else:
                CalculatedOddsRatiosComponent = (A*D)/(B*C)      
            CalculatedOddsRatios.append(CalculatedOddsRatiosComponent)
    return CalculatedOddsRatios


def OddsRatiosByRegression(Candidate) :
    "Calculate Odds ratio by using logistic regression"
    Factors = len(Candidate)
    CalculatedOddsRatios = []
    # perform regression for any combination of risk factors/outcome
    # organize results in descending order such that 
    # Rn/Rn-1 is first and R2/R1 is last
    for OutcomeIndex in reversed(range(Factors)):
        for ValuesIndex in reversed(range(OutcomeIndex)):
            try:
                ValuesX = numpy.asarray(Candidate[ValuesIndex])
                ValuesXReshaped = ValuesX[:,numpy.newaxis]
                ValuesY = numpy.asarray(Candidate[OutcomeIndex])
                LogisticRegressionObject = LogisticRegression()
                LogisticRegressionObject.fit(ValuesXReshaped, ValuesY)
                LogCalcualtedOddsRatio = LogisticRegressionObject.coef_[0][0]
                CalcualtedOddsRatio = numpy.exp(LogCalcualtedOddsRatio)
            except ValueError:
                # In the event that we end up with a population that all have
                # a given RF, the regression will fail.
                CalcualtedOddsRatio = None
            CalculatedOddsRatios.append(CalcualtedOddsRatio)
    return CalculatedOddsRatios



def CalculateStatistics(Candidate, OddsRatiosCalculationFunction):
    "Return statistics for a solution"
    
    CalculatedProbabilities = [ numpy.mean(ParameterVector) for ParameterVector in Candidate ]
    CalculatedDivisionRatios = OddsRatiosByDefinition(Candidate,True)
    CalculatedOddsRatios = OddsRatiosCalculationFunction(Candidate)
    return CalculatedProbabilities, CalculatedOddsRatios, CalculatedDivisionRatios

def GenerateSample(random, args):
    "Generate a single solution"
    
    # The generated sample will be a series of binary
    # values corresponding to whether each participant has
    # Risk factor i. Each risk factor will be generated using
    # Bernoulli distribution with the given probability.

    N = args.get('OutputPopualtionSize', 100)
    Probabilities = args.get('Probabilities', [])
    Candidate =  [ random.binomial(1,abs(Probability),N) for Probability in Probabilities]
    return Candidate


def AbsoluteDiff(Value1,Value2):
    " return absolute difference "
    Diff = abs(Value1 - Value2)
    return Diff

def RelativeDiff(Value1,Value2):
    " return absolute difference "
    Diff = abs((Value1 - Value2)/Value2)
    return Diff

def DefaultRandomSeedFunction(Input):
    "A default random seed function that returns identity"
    return Input




@inspyred.ec.evaluators.evaluator
def EvaluateSample(Candidate, args):
    Probabilities = args.get('Probabilities', [])
    OddsRatios = args.get('OddsRatios', [])
    DivisionRatios = args.get('DivisionRatios', [])
    ProbabilityWeightsList = args.get('ProbabilityWeightsList', [])
    OddsRatiosWeightsList = args.get('OddsRatiosWeightsList', [])
    DivisionRatiosWeightsList = args.get('DivisionRatiosWeightsList', [])
    
    FitnessDiffFunction = args.get('FitnessDiffFunction', AbsoluteDiff)
    OddsRatiosCalculationFunction = args.get('OddsRatiosCalculationFunction',OddsRatiosByDefinition)  
    (CalcualtedProbabilities, CalculatedOddsRatios, CalculatedDivisionRatios) = CalculateStatistics(Candidate, OddsRatiosCalculationFunction)
    Fitness = 0
    for CalcualtedProbability, Probability, ProbabilityWeight in zip(CalcualtedProbabilities, Probabilities, ProbabilityWeightsList):
        if Probability >=0:
            Fitness += FitnessDiffFunction(CalcualtedProbability,Probability)*ProbabilityWeight
    for CalcualtedOddsRatio, CalculatedDivisionRatio, OddsRatio, DivisionRatio, OddsRatioWeight, DivisionRatiosWeight in zip(CalculatedOddsRatios, CalculatedDivisionRatios, OddsRatios, DivisionRatios, OddsRatiosWeightsList, DivisionRatiosWeightsList):
        if OddsRatio is not None:
            if CalcualtedOddsRatio is None or CalcualtedOddsRatio!=CalcualtedOddsRatio:
                # In the event that we end up with a population that all have
                # a given RF, the regression will fail. In that case, make
                # the fitness very high.  Also check division by zero
                Fitness += 999999
            else:
                Fitness += FitnessDiffFunction(CalcualtedOddsRatio,OddsRatio)*OddsRatioWeight

        if DivisionRatio is not None:
            if CalculatedDivisionRatio is None or CalculatedDivisionRatio!=CalculatedDivisionRatio:
                # handle anomaly just like OddsRatio
                Fitness += 999999
            else:
                Fitness += FitnessDiffFunction(CalculatedDivisionRatio,DivisionRatio)*DivisionRatiosWeight

    return Fitness



@inspyred.ec.variators.crossover
def Crossover(random, Mom, Dad, args):
    "Crossover round swaps between two tournament solutions"
    Brother = copy.deepcopy(Dad)
    Sister = copy.deepcopy(Mom)
    N = len(Dad[0])
    NumberOfVectors = len(Dad)
    for VectorEnum in range(NumberOfVectors):
        # there is 50% chance of swap between elements
        Mask = random.choice(2,N)
        numpy.putmask(Brother[VectorEnum], Mask, Mom[VectorEnum])
        numpy.putmask(Sister[VectorEnum], Mask, Dad[VectorEnum])
    return Brother,Sister

@inspyred.ec.variators.mutator
def Mutator1(random, Candidate, args):
    "Mutate bits to add some variation"
    # Note that this mutator does not preserve initial distribution
    MutationRate = args['MutationRate1']
    Mutated = copy.deepcopy(Candidate)
    VectorSize = len(Candidate[0])
    NumberOfVectors = len(Candidate)
    for VectorEnum in range(NumberOfVectors):
        # there is a change for flipping each element
        Mask = random.binomial(1,MutationRate,VectorSize)
        numpy.putmask(Mutated[VectorEnum], Mask, 1-Candidate[VectorEnum])
    return Mutated

@inspyred.ec.variators.mutator
def Mutator2(random, Candidate, args):
    "Mutate swaps to add some variation"
    MutationRate = args['MutationRate2']
    Mutated = copy.deepcopy(Candidate)
    VectorSize = len(Candidate[0])
    NumberOfVectors = len(Candidate)
    for VectorEnum in range(NumberOfVectors):
        # swap two characteristics in the vector
        if random.random()<MutationRate:
            SwapEnum1 = random.randint(0,VectorSize)
            SwapEnum2 = random.randint(0,VectorSize)
            Mutated[VectorEnum][SwapEnum1]=Candidate[VectorEnum][SwapEnum2]
            Mutated[VectorEnum][SwapEnum2]=Candidate[VectorEnum][SwapEnum1]
    return Mutated

@inspyred.ec.variators.mutator
def Mutator3(random, Candidate, args):
    "Mutate by re-roll of random numbers according to initial probability"
    # This mutator attempts to preserve initial distribution on average
    # since the newly generated vector uses the original distributions
    # and mutations arrive from it
    MutationRate = args['MutationRate3']   
    NewlyGenerated = GenerateSample(random, args)
    Mutated = copy.deepcopy(Candidate)
    NumberOfVectors = len(Candidate)
    for VectorEnum in range(NumberOfVectors):
        # borrow the new record from newly generated sample
        if random.random()<MutationRate:
            Mutated[VectorEnum]=NewlyGenerated[VectorEnum]
    return Mutated


# Default optimization parameters
class OptimizationParametersClass():
    "Structure holding EC optimization parameters"
    def __init__(self): 
        "Initialize to these defaults"     
        self.PopulationOfPopulationsSize=100
        self.MaxEvaluations=100000
        self.NumberSelected=100
        self.MutationRate1=0
        self.MutationRate2=0.02
        self.MutationRate3=0.002
        self.NumberOfElites=2
        self.FitnessDiffFunction = AbsoluteDiff
        self.OddsRatiosCalculationFunction = OddsRatiosByDefinition
        # Order of probabilities:  R1, R2, ... Rn where Rn is the Output
        # Order of Odds ratios: Rn/Rn-1, Rn/Rn-2...Rn-1/Rn-2, Rn-1/Rn-2...R2/R1
        # where missing elements are marked negative for probabilities
        # or are marked with  None in Odds ratio     
        self.OutputPopualtionSize = None
        self.Probabilities = None
        self.OddsRatios = None
        self.DivisionRatios = None
        self.ProbabilityWeights = None
        self.OddsRatiosWeights = None
        self.OutputPopulationFileName = None
        self.RandomSeed = None
        
        self.ProbabilityWeights = None
        self.OddsRatiosWeights = None
        self.DivisionRatiosWeights = None
        
        self.DaskClientAddress = None
        self.OutputPopulationFileNamePattern = None
        self.OutputPopulationIndexTuple = None
        self.NumberOfRepetitions = None
        self.RandomSeedFunction = None
        self.AggregateOutputFileName = None
        self.RepetitionNumber = None
        self.WorkingDirectory = None
        self.TimesToRetryFileWrite = None
        self.SkipPhases = []

        self.PlotFileName = None
        self.PlotImageFileName = None
        self.PlotFileWorksOffline = True
        self.HistogramNumberOfBins = 10
        self.HistogramPlotTitle = ""
        self.ProbabilitiesTitles = None
        self.OddsRatiosTitles = None
        self.LaunchPlotInBrowser = False
        
        self.MaxIterationsForInitialProbabilityOptimization = None
        self.ToleranceForInitialProbabilityOptimization  = None
        self.IterationNumberForInitialProbabilityOptimization = 0
        self.IterationStrategy = None






    def ExtendParameters(self):     
        "Load problem parameters"
        if type(self.ProbabilityWeights) == type([]):
            self.ProbabilityWeightsList = self.ProbabilityWeights
        else:
            self.ProbabilityWeightsList = [self.ProbabilityWeights]*len(self.Probabilities)         
        if type(self.OddsRatiosWeights) == type([]):
            self.OddsRatiosWeightsList = self.OddsRatiosWeights
        else:
            self.OddsRatiosWeightsList = [self.OddsRatiosWeights]*len(self.OddsRatios)
            
        if type(self.DivisionRatiosWeights) == type([]):
            self.DivisionRatiosWeightsList = self.DivisionRatiosWeights
        else:
            self.DivisionRatiosWeightsList = [self.DivisionRatiosWeights]*len(self.DivisionRatios)

        if self.WorkingDirectory == None:
            self.WorkingDirectoryResolved = os.getcwd()
        else:
            self.WorkingDirectoryResolved = self.WorkingDirectory
        if type(self.ProbabilitiesTitles) == type([]):
            self.ProbabilitiesTitlesResolved = self.ProbabilitiesTitles
        else:
            self.ProbabilitiesTitlesResolved = [self.ProbabilitiesTitles]*len(self.Probabilities)
        for (Enum,Entry) in enumerate(self.ProbabilitiesTitlesResolved):
            if Entry is None:
                self.ProbabilitiesTitlesResolved[Enum] = 'RF%i'%Enum
        if type(self.OddsRatiosTitles) == type([]):
            self.OddsRatiosTitlesResolved = self.OddsRatiosTitles
        else:
            self.OddsRatiosTitlesResolved = [self.OddsRatiosTitles]*len(self.OddsRatios)
        OddsIndex = 0
        for (OutcomeIndex,OutcomeName) in reversed(list(enumerate(self.ProbabilitiesTitlesResolved))):
            for (ValuesIndex,ValueName) in reversed(list(enumerate(self.ProbabilitiesTitlesResolved[:OutcomeIndex]))):
                if self.OddsRatiosTitlesResolved[OddsIndex] is None:
                    self.OddsRatiosTitlesResolved[OddsIndex] = OutcomeName+':'+ValueName
                OddsIndex = OddsIndex + 1



    def CalculateFileName(self, FileNamePattern ):
        "return file name from combining pattern and IndexTuple"
        TupleSizeToUse = len(self.ExpandedOutputPopulationIndexTuple)
        while TupleSizeToUse>0:
            try:
                # try to use the pattern and index data
                ReturnedFileName = FileNamePattern % (self.ExpandedOutputPopulationIndexTuple[0:TupleSizeToUse])
                # if not exception, just break the while loop
                break
            except:
                # if not successful ue a smaller tuple ignoring the end
                TupleSizeToUse = TupleSizeToUse - 1

        if TupleSizeToUse == 0:
            # Finally use only the pattern
            ReturnedFileName = FileNamePattern
        return ReturnedFileName

        
    def LoadYamlData(self, YamlFileName):
        "Loads data from configuration file to the system"
        YamlFile = open(YamlFileName)
        DataDict = yaml.safe_load(YamlFile)
        YamlFile.close()
        YamlReplaceDict   = { ('OptimizationParameters', 'FitnessDiffFunction', 'AbsoluteDiff'):  AbsoluteDiff , 
                              ('OptimizationParameters', 'FitnessDiffFunction', 'RelativeDiff'):  RelativeDiff , 
                              ('OptimizationParameters', 'OddsRatiosCalculationFunction', 'OddsRatiosByDefinition'):  OddsRatiosByDefinition , 
                              ('OptimizationParameters', 'OddsRatiosCalculationFunction', 'OddsRatiosByRegression'):  OddsRatiosByRegression , 
                              ('ExecutionParameters', 'RandomSeedFunction', 'DefaultRandomSeedFunction'):  DefaultRandomSeedFunction ,         
                              }
            
        for (Category,CategoryDict) in DataDict.items():
            for (Item, ItemValue) in CategoryDict.items():
                NewItemValue = ItemValue
                # Take care of None values first
                if ItemValue == 'None':
                    NewItemValue = None
                elif type(ItemValue) == type([]):
                    # None can be in a list
                    NewItemValue = []
                    for Entry in ItemValue:
                        if Entry == 'None':
                            NewItemValue.append(None)
                        else:
                            NewItemValue.append(Entry)
                # check for replacements 
                if type(NewItemValue) == type('') and (Category,Item,NewItemValue) in YamlReplaceDict:
                    NewItemValue = YamlReplaceDict[(Category,Item,NewItemValue)]
                # Finally update self
                if Item in dir(self):
                    setattr(self,Item,NewItemValue)
                else:
                    raise ValueError, 'Invalid Yaml Parameter:' + str((Category,Item,ItemValue))
        self.ExtendParameters()
        return self


def ApplyEvolutionalryComputation(OptimizationParameters):
    "Define and Run the Evolutionary Computation"
    
    class NumpyRandomWrapper(numpy.random.RandomState):
        def __init__(self, seed=None):
            super(NumpyRandomWrapper, self).__init__(seed)
            
        def sample(self, population, k):
            return self.choice(population, k, replace=False)
            
        def random(self):
            return self.random_sample()    
        
    RandomGeneratorToUse = NumpyRandomWrapper(OptimizationParameters.RandomSeed)

    # if user defines input as a list use it, otherwise expand to list
    SolutionObject = inspyred.ec.EvolutionaryComputation(RandomGeneratorToUse)
    SolutionObject.selector = inspyred.ec.selectors.tournament_selection
    SolutionObject.variator = [Crossover, Mutator1, Mutator2, Mutator3]
    SolutionObject.replacer = inspyred.ec.replacers.generational_replacement
    SolutionObject.observer = inspyred.ec.observers.stats_observer
    SolutionObject.terminator = inspyred.ec.terminators.evaluation_termination

    FinalPop = SolutionObject.evolve(generator=GenerateSample,
                             evaluator=EvaluateSample,
                             bounder=inspyred.ec.DiscreteBounder([0, 1]),
                             pop_size=OptimizationParameters.PopulationOfPopulationsSize,
                             max_evaluations=OptimizationParameters.MaxEvaluations,
                             num_selected=OptimizationParameters.NumberSelected,
                             MutationRate1=OptimizationParameters.MutationRate1,
                             MutationRate2=OptimizationParameters.MutationRate2,
                             MutationRate3=OptimizationParameters.MutationRate3,
                             num_elites=OptimizationParameters.NumberOfElites,
                             OutputPopualtionSize=OptimizationParameters.OutputPopualtionSize,
                             maximize=False,
                             OddsRatiosCalculationFunction = OptimizationParameters.OddsRatiosCalculationFunction,
                             FitnessDiffFunction = OptimizationParameters.FitnessDiffFunction,
                             Probabilities=OptimizationParameters.Probabilities,  
                             OddsRatios=OptimizationParameters.OddsRatios, 
                             DivisionRatios = OptimizationParameters.DivisionRatios,
                             ProbabilityWeightsList = OptimizationParameters.ProbabilityWeightsList,
                             OddsRatiosWeightsList = OptimizationParameters.OddsRatiosWeightsList,
                             DivisionRatiosWeightsList = OptimizationParameters.DivisionRatiosWeightsList
                             )
    # Sort and print the best individual, who will be at index 0.
    FinalPop.sort(reverse=True)
    OutputPopulationFileName = OptimizationParameters.CalculateFileName(OptimizationParameters.OutputPopulationFileNamePattern)
    print 'Final population save in the file: ' + OutputPopulationFileName
    ColumnNames = ['Rf'+str(Enum+1) for Enum in range(len(OptimizationParameters.Probabilities))]
    NumberOfWriteRetries = 0
    while True:
        OutputDataFrame = pandas.DataFrame(FinalPop[0].candidate).transpose()
        OutputDataFrame.to_csv(OutputPopulationFileName , header = ColumnNames, index_label ='ID')
        # check that file was written
        FileExisits = os.path.isfile(OutputPopulationFileName)
        if not FileExisits:
            if NumberOfWriteRetries<OptimizationParameters.TimesToRetryFileWrite:
                NumberOfWriteRetries = NumberOfWriteRetries + 1
                print 'Retry #i of write file since could not locate file: %s' + (NumberOfWriteRetries, OutputPopulationFileName)
                continue
            else:
                # dump file for possible future inspection
                OutputDataFrame.to_pickle(OutputPopulationFileName+'.pickle')
                print "ERROR - while writing file " + OutputPopulationFileName
        else:
            # file found - break out of loop
            break
    (CalculatedProbabilities, CalculatedOddsRatios, CalculatedDivisionRatios) = CalculateStatistics(FinalPop[0].candidate, OptimizationParameters.OddsRatiosCalculationFunction)
    print 'Final population saved in the file: ' + OutputPopulationFileName
    print 'The probabilities requested: ' + str(OptimizationParameters.Probabilities)
    print 'Final probabilities Reached: ' + str(CalculatedProbabilities)
    print 'The odds ratios requested: ' + str(OptimizationParameters.OddsRatios)
    print 'Final odds ratios reached: ' +str(CalculatedOddsRatios)
    print 'The division ratios requested: ' + str(OptimizationParameters.DivisionRatios)
    print 'Final division ratios reached: '  +str(CalculatedDivisionRatios)
    return (CalculatedProbabilities, CalculatedOddsRatios, CalculatedDivisionRatios)



def CalcExpandedOutputPopulationIndexTuple (OptimizationParametersTemplate, OptimizationParameters, RepetitionEnum):
    "Expand tuple used for file name"
    # do it only if the expanded attribute does not  exist already
    if 'ExpandedOutputPopulationIndexTuple' not in dir(OptimizationParameters):
        if OptimizationParametersTemplate.OutputPopulationIndexTuple is None:
            TupleList= (OptimizationParameters.IterationNumberForInitialProbabilityOptimization, RepetitionEnum)
        else:
            TupleList = []
            for Entry in OptimizationParametersTemplate.OutputPopulationIndexTuple:
                TupleElement = getattr(OptimizationParametersTemplate, Entry)
                TupleList.append(TupleElement)
        OptimizationParameters.ExpandedOutputPopulationIndexTuple = tuple(TupleList)
    return OptimizationParameters



@dask.delayed
def DelayedLaunchJob(OptimizationParametersTemplate,RepetitionEnum):
    "Lazy launch job"    
    OptimizationParameters = copy.deepcopy(OptimizationParametersTemplate)
    if OptimizationParametersTemplate.RandomSeedFunction is not None:
        OptimizationParameters.RandomSeed = OptimizationParametersTemplate.RandomSeedFunction(RepetitionEnum)
    OptimizationParameters.RepetitionEnum = RepetitionEnum
    # Determine the file name elements
    CalcExpandedOutputPopulationIndexTuple (OptimizationParametersTemplate, OptimizationParameters, RepetitionEnum)
                
    if 'EC' in OptimizationParametersTemplate.SkipPhases:
        # skip computation and report zero results
        ReturnValue = ( [0]*len(OptimizationParametersTemplate.Probabilities), [0]*len(OptimizationParametersTemplate.OddsRatios), [0]*len(OptimizationParametersTemplate.OddsRatios) )
    else:
        os.chdir(OptimizationParameters.WorkingDirectoryResolved)
        ReturnValue = ApplyEvolutionalryComputation(OptimizationParameters)
    return (RepetitionEnum, OptimizationParameters, ReturnValue)
    
@dask.delayed
def DelayedReduce(JobArray, OptimizationParameters) :
    "Reduce the array"
    ResultsArray = []
    os.chdir(OptimizationParameters.WorkingDirectoryResolved)
    if 'Summarize' in OptimizationParametersTemplate.SkipPhases:
        # skip computation and report zero results, do not output file
        ResultsArray = []
    else:
        for (RepetitionEnum, OptimizationParameters, (CalculatedProbabilities, CalculatedOddsRatios, CalculatedDivisionRatios))  in JobArray:
            ResultsArray.append( ['Enum:'] + [RepetitionEnum] + ['Probabilities: '] + CalculatedProbabilities  + ['CalculatedOddsRatios: '] +  CalculatedOddsRatios + ['CalculatedDivisionRatios: '] +  CalculatedDivisionRatios)
        CalcExpandedOutputPopulationIndexTuple (OptimizationParameters, OptimizationParameters, 0)
        FileNameAggregate = OptimizationParameters.CalculateFileName(OptimizationParameters.AggregateOutputFileName)
        pandas.DataFrame(ResultsArray).to_csv(FileNameAggregate , header = False, index = False)
    return  (JobArray, OptimizationParameters, ResultsArray)

@dask.delayed
def DelayedPlot(ReduceOutput) :
    "Show the summary file as a plot"
    (JobArray, OptimizationParameters, ResultsArray) = ReduceOutput
    os.chdir(OptimizationParameters.WorkingDirectoryResolved)
    PlotNames = []
    if 'Plot' in OptimizationParameters.SkipPhases:
        PlotNames = None
    else:
        CalcExpandedOutputPopulationIndexTuple (OptimizationParameters, OptimizationParameters, 0)
        FileNameAggregate = OptimizationParameters.CalculateFileName(OptimizationParameters.AggregateOutputFileName)
        # extract data from file
        DataFrame = pandas.read_csv(filepath_or_buffer = FileNameAggregate, header = None)
        ResultsEnums = []
        ResultsProbabilities = []
        ResultsOddsRatios = []
        ResultsDivisionRatios = []
        for (RowIndex,Row) in DataFrame.iterrows():
            ListRow = list(Row)
            EnumDivider = ListRow.index('Enum:')
            ProbabilitiesDivider = ListRow.index('Probabilities: ')
            OddsRatiosDivider = ListRow.index('CalculatedOddsRatios: ')
            DivisionRatiosDivider = ListRow.index('CalculatedDivisionRatios: ')
            ResultsEnums.append(int(ListRow[EnumDivider+1:ProbabilitiesDivider][0]))
            ResultsProbabilities.append(ListRow[ProbabilitiesDivider+1:OddsRatiosDivider])
            ResultsOddsRatios.append(ListRow[OddsRatiosDivider+1:DivisionRatiosDivider])
            ResultsDivisionRatios.append(ListRow[DivisionRatiosDivider+1:])
            
        ResultsProbabilitiesExpanded = reduce(operator.add,  ResultsProbabilities , []) 
        ResultsOddsRatiosExpanded = reduce(operator.add,  ResultsOddsRatios , [])
        ResultsDivisionRatiosExpanded = reduce(operator.add,  ResultsDivisionRatios , [])

        if OptimizationParameters.PlotFileName is not None:
            try:
                FileNameHTML = OptimizationParameters.CalculateFileName(OptimizationParameters.PlotFileName)
                Title = 'Populations Generated Iteration %i' % ( OptimizationParameters.IterationNumberForInitialProbabilityOptimization)
                bokeh.io.output_file(filename = FileNameHTML, title=Title, mode='inline')               
            except:
                print ('Could not initialize html file')

        # Show probabilities plot
        PlotArray = []
        if OptimizationParameters.HistogramPlotTitle[0] != None:
            
            MyHover1 = bokeh.models.HoverTool(
                tooltips=[
                    ( 'Probability Name', '@ProbabilityName{%s}'),
                    ( 'Probability', '@Probability' ),
                    ( 'Simulation Number', '@SimulationNumber' ),  

                ],
                formatters={
                    'ProbabilityName' : 'printf',   
                    'Probability' : 'numeral',   
                    'SimulationNumber' : 'numeral', 
                },
                point_policy="follow_mouse"
            )            
            
            Plot = bokeh.plotting.figure(title = OptimizationParameters.HistogramPlotTitle[0], 
                                  x_axis_label = 'Probability Name', 
                                  y_axis_label = 'Probability Value', 
                                  tools = ['save',MyHover1], 
                                  x_range = OptimizationParameters.ProbabilitiesTitlesResolved , 
                                  y_range = bokeh.models.Range1d(-0.1, 1.1))
            Plot.xaxis.major_label_orientation = "vertical"
            
            BarSource = bokeh.models.ColumnDataSource(dict(
                    ProbabilityName = OptimizationParameters.ProbabilitiesTitlesResolved, 
                    Probability = [abs(Entry) for Entry in OptimizationParameters.Probabilities] , 
                    FillColor = [['Blue','Red'][Entry<0] for Entry in OptimizationParameters.Probabilities],
                    SimulationNumber = [None]*len(OptimizationParameters.ProbabilitiesTitlesResolved)
                    ))

            CircleSource = bokeh.models.ColumnDataSource(dict(
                    ProbabilityName = reduce(operator.add, [OptimizationParameters.ProbabilitiesTitlesResolved]*len(ResultsEnums)), 
                    Probability = ResultsProbabilitiesExpanded ,
                    SimulationNumber = reduce(operator.add, [ [Entry]*len(OptimizationParameters.Probabilities) for Entry in ResultsEnums ] , []) ,
                    ))

            Plot.vbar(source = BarSource, x='ProbabilityName', width = 0.7, bottom = 0 , top = 'Probability', fill_color = 'FillColor', line_color='black')
            Plot.circle(source = CircleSource, x='ProbabilityName',y='Probability', fill_color = 'yellow', fill_alpha = 0, line_width = 1.5, line_color='purple', size = 5)
            
            PlotArray.append(Plot)


        if OptimizationParameters.HistogramPlotTitle[1] != None:
            MyHover2 = bokeh.models.HoverTool(
                tooltips=[
                    ( 'Odds Ratio Name', '@OddsRatioName{%s}'),
                    ( 'Odds Ratio', '@OddsRatio' ),
                    ( 'Simulation Number', '@SimulationNumber' ),  
                ],
                formatters={
                    'OddsRatioName' : 'printf',   
                    'OddsRatio' : 'numeral',   
                    'SimulationNumber' : 'numeral', 
                },
                point_policy="follow_mouse"
            )
            
            Plot = bokeh.plotting.figure(title = OptimizationParameters.HistogramPlotTitle[1], 
                                  x_axis_label = 'Odds Ratio Parameters', 
                                  y_axis_label = 'Odds Ratio Value', 
                                  tools = ['save',MyHover2], 
                                  x_range = OptimizationParameters.OddsRatiosTitlesResolved , 
                                  y_range = bokeh.models.Range1d(-0.1, 1.1*max(ResultsOddsRatiosExpanded)))
            Plot.xaxis.major_label_orientation = "vertical"
            
            BarSource = bokeh.models.ColumnDataSource(dict(
                    OddsRatioName = OptimizationParameters.OddsRatiosTitlesResolved, 
                    OddsRatio = OptimizationParameters.OddsRatios , 
                    FillColor = ['Green' for Entry in OptimizationParameters.OddsRatios] ,
                    SimulationNumber = [None]*len(OptimizationParameters.OddsRatiosTitlesResolved)
                    ))

            CircleSource = bokeh.models.ColumnDataSource(dict(
                    OddsRatioName = reduce(operator.add,[OptimizationParameters.OddsRatiosTitlesResolved]*len(ResultsEnums),[]), 
                    OddsRatio = ResultsOddsRatiosExpanded,
                    SimulationNumber = reduce(operator.add, [ [Entry]*len(OptimizationParameters.OddsRatios) for Entry in ResultsEnums ] , []) ,
                    ))
            
            Plot.vbar(source = BarSource, x='OddsRatioName', width = 0.7, bottom = 0 , top = 'OddsRatio', fill_color = 'FillColor', line_color='black')
            Plot.circle(source = CircleSource, x='OddsRatioName',y='OddsRatio', fill_color = 'yellow', fill_alpha = 0, line_width = 1.5, line_color='purple', size = 5)
            
            PlotArray.append(Plot)

        if OptimizationParameters.HistogramPlotTitle[2] != None:
            
            MyHover3 = bokeh.models.HoverTool(
                tooltips=[
                    ( 'Division Ratio Name', '@DivisionRatioName{%s}'),
                    ( 'Division Ratio', '@DivisionRatio' ),
                    ( 'Simulation Number', '@SimulationNumber' ),  
                ],
                formatters={
                    'DivisionRatioName' : 'printf',   
                    'DivisionRatio' : 'numeral',   
                    'SimulationNumber' : 'numeral', 
                },
                point_policy="follow_mouse"
            )
            
            Plot = bokeh.plotting.figure(title = OptimizationParameters.HistogramPlotTitle[2], 
                                  x_axis_label = 'Division Ratio Parameters', 
                                  y_axis_label = 'Division Ratio Value', 
                                  tools = ['save',MyHover3], 
                                  x_range = OptimizationParameters.OddsRatiosTitlesResolved , 
                                  y_range = bokeh.models.Range1d(-0.1, 1.1*max(ResultsDivisionRatiosExpanded)))
            Plot.xaxis.major_label_orientation = "vertical"
            
            BarSource = bokeh.models.ColumnDataSource(dict(
                    DivisionRatioName = OptimizationParameters.OddsRatiosTitlesResolved, 
                    DivisionRatio = OptimizationParameters.DivisionRatios , 
                    FillColor = ['Green' for Entry in OptimizationParameters.DivisionRatios] ,
                    SimulationNumber = [None]*len(OptimizationParameters.OddsRatiosTitlesResolved)
                    ))

            CircleSource = bokeh.models.ColumnDataSource(dict(
                    DivisionRatioName = reduce(operator.add,[OptimizationParameters.OddsRatiosTitlesResolved]*len(ResultsEnums),[]), 
                    DivisionRatio = ResultsDivisionRatiosExpanded,
                    SimulationNumber = reduce(operator.add, [ [Entry]*len(OptimizationParameters.DivisionRatios) for Entry in ResultsEnums ] , []) ,
                    ))

            
            Plot.vbar(source = BarSource, x='DivisionRatioName', width = 0.7, bottom = 0 , top = 'DivisionRatio', fill_color = 'FillColor', line_color='black')
            Plot.circle(source = CircleSource, x='DivisionRatioName',y='DivisionRatio', fill_color = 'yellow', fill_alpha = 0, line_width = 1.5, line_color='purple', size = 5)
            
            PlotArray.append(Plot)

        def PlotHistogram(PlotArray, PlotTitle, AxisTitles, ReferenceValues, Values):
            "Plot Histogram for unset values"

            MyHover4 = bokeh.models.HoverTool(
                tooltips=[
                    ( 'Object', '@Object{%s}'),
                    ( 'From', '@EdgeLeft'),
                    ( 'To', '@EdgeRight' ),
                    ( 'Count', '@Count' ),  
                    ( 'Mean', '@ValueMean' ),  
                    ( 'STD', '@ValueSTD' ),  
                ],
                formatters={
                    'Object' : 'printf',
                    'EdgeLeft' : 'numeral',   
                    'EdgeRight' : 'numeral',   
                    'Count' : 'numeral', 
                    'ValueMean' : 'numeral', 
                    'ValueSTD' : 'numeral', 
                },
                point_policy="follow_mouse"
            )
            
            ResultValuesMeans = []
            ResultValuesSTDs = []

            for (ValueEnum, ReferenceValue) in enumerate(ReferenceValues):
                ExtractedValues = [ValuesList[ValueEnum] for ValuesList in Values]
                ValuesMean = numpy.mean(ExtractedValues)
                ResultValuesMeans.append(ValuesMean)
                ValuesSTD = numpy.std(ExtractedValues)        
                ResultValuesSTDs.append(ValuesSTD)        
                if PlotTitle is not None:
                    if ReferenceValue < 0 or ReferenceValue is None:       
                        (HistogramBarValues, BinEdges) = numpy.histogram(ExtractedValues, density=False, bins=OptimizationParameters.HistogramNumberOfBins)                 
    
                        HeightRef = max(HistogramBarValues)
                        Plot = bokeh.plotting.figure(title = PlotTitle, 
                                             x_axis_label = 'Range of ' + AxisTitles[ValueEnum] , 
                                             y_axis_label = 'Occurrences', 
                                             tools = ['save', MyHover4], 
                                             x_range = bokeh.models.Range1d(0.9 * BinEdges[0], 1.1*BinEdges[-1]) , 
                                             y_range = bokeh.models.Range1d(-0.1*HeightRef, 1.1*HeightRef))
                        Plot.xaxis.major_label_orientation = "vertical"

                        HistogramSources = bokeh.models.ColumnDataSource(dict(
                            Object = ['Histogram Bar' ]*len(HistogramBarValues),
                            EdgeLeft = BinEdges[:-1],
                            EdgeRight = BinEdges[1:],
                            Count = HistogramBarValues,
                            MeanValue = (BinEdges[1:]+BinEdges[:-1])/2,
                            ProbabilitySTD = [None]*len(HistogramBarValues),
                            ))
                        
                        SummaryStatistics = bokeh.models.ColumnDataSource(dict(
                            Object = ['Summary Statistics' ],
                            ValueMean = [ValuesMean],
                            ValueSTD = [ValuesSTD],
                            EdgeLeft = [ValuesMean-ValuesSTD],
                            EdgeRight = [ValuesMean+ValuesSTD],
                            Count = [len(ExtractedValues)] ,
                            ))

                        Plot.quad(source = HistogramSources, bottom = 0, top = 'Count' , left = 'EdgeLeft' , right = 'EdgeRight', fill_color='cyan', line_color = 'blue' )
                        Plot.quad(source = SummaryStatistics, bottom = -0.08*HeightRef, top = -0.02*HeightRef , left = 'EdgeLeft' , right = 'EdgeRight', fill_color='red' )
                        Plot.vbar(source = SummaryStatistics, bottom = -0.1*HeightRef, top = 1.1*HeightRef , width=(BinEdges[1]-BinEdges[0])*0.2, x = 'ValueMean', fill_color = 'black')

                        PlotArray.append(Plot)
            return (ResultValuesMeans, ResultValuesSTDs)
                
        


        # probability Histogram for each negative probability
        ResultProbabilitiesMeans, ResultProbabilitiesSTDs = PlotHistogram(PlotArray, OptimizationParameters.HistogramPlotTitle[3], OptimizationParameters.ProbabilitiesTitlesResolved, OptimizationParameters.Probabilities, ResultsProbabilities)

        # probability Histogram for each negative probability
        ResultOddsRatiosMeans, ResultOddsRatiosSTDs = PlotHistogram(PlotArray, OptimizationParameters.HistogramPlotTitle[4], OptimizationParameters.OddsRatiosTitlesResolved, OptimizationParameters.OddsRatios, ResultsOddsRatios)

        # probability Histogram for each negative probability
        ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs = PlotHistogram(PlotArray, OptimizationParameters.HistogramPlotTitle[5], OptimizationParameters.OddsRatiosTitlesResolved, OptimizationParameters.DivisionRatios, ResultsDivisionRatios)



        if PlotArray != []:
            Layout = bokeh.layouts.column(PlotArray)

            if OptimizationParameters.PlotImageFileName is not None:
                try:
                    ImageFileName = OptimizationParameters.CalculateFileName(OptimizationParameters.PlotImageFileName)
                    bokeh.io.export_png(Layout,filename=ImageFileName)
                    PlotNames.append(ImageFileName)
                except:
                    print ('Could not export png image file')
            if OptimizationParameters.PlotFileName is not None:
                try:
                    bokeh.io.save(obj = Layout)     
                    PlotNames.append(FileNameHTML)                    
                except:
                    print ('Could not export html file')
                if OptimizationParameters.LaunchPlotInBrowser:
                    webbrowser.open(FileNameHTML)
    
        return (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs, PlotNames)
          





def LaunchDistributedSimulations(OptimizationParametersTemplate):
    "Launch simulations in a distributed manner"
    
    if OptimizationParametersTemplate.DaskClientAddress is None:
        DaskClient = None # use basic client
    elif OptimizationParametersTemplate.DaskClientAddress == 0:
        # if 0 specified, create local distributed client
        DaskClient = dask.distributed.Client()     
    else:
        # connect to existing running server
        DaskClient = dask.distributed.Client(OptimizationParametersTemplate.DaskClientAddress)       

    def ComputeError(ResultProbabilities,ExpectedProbabilities):
        "Calculates error norm only for elements that can change"
        Sum = 0
        for (Enum, ExpectedProbability) in enumerate(ExpectedProbabilities):
            if ExpectedProbability<0:
                # remember that expected probabilities that can change are
                # already negative numbers
                Sum = Sum + (ResultProbabilities[Enum]+ExpectedProbability)**2
        RetVal = Sum**0.5
        return RetVal
    
    
    def CalculateNewProbability(Template, ResultProbabilities):
        "Calculates error norm only for elements that can change"
        ExpectedProbabilities = Template.Probabilities
        RetVal = []
        if Template.IterationStrategy is None:
            # The case of drift calculation
            for (Enum, ExpectedProbability) in enumerate(ExpectedProbabilities):
                if ExpectedProbability<0:
                    RetVal.append(-ResultProbabilities[Enum])
                else:
                    RetVal.append(ExpectedProbabilities[Enum])
        else:
            RetVal = Template.IterationStrategy[Template.IterationNumberForInitialProbabilityOptimization]
        Template.Probabilities = RetVal
        return RetVal

    Template = OptimizationParametersTemplate
    
    AdjustableProbabilityError = Template.ToleranceForInitialProbabilityOptimization *2
    while True:
        print 'Executing Iteration #%i \n Probabilities = %s \n Odds Ratios = %s \n Division Ratios = %s'% (Template.IterationNumberForInitialProbabilityOptimization, str(Template.Probabilities), str(Template.OddsRatios), str(Template.DivisionRatios) )
        JobArray = [ DelayedLaunchJob(Template,RepetitionEnum) for RepetitionEnum in range(Template.NumberOfRepetitions)]      
        ReduceOutput = DelayedReduce (JobArray, Template)
        PlotOutput = DelayedPlot(ReduceOutput)
        (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs, PlotNames) = PlotOutput.compute()
        AdjustableProbabilityError = ComputeError(ResultProbabilitiesMeans, Template.Probabilities)
        # construct a new template
        Template = copy.deepcopy(Template)
        Template.IterationNumberForInitialProbabilityOptimization = Template.IterationNumberForInitialProbabilityOptimization + 1
        print 'After simulation, adjustable probability Error was: %g allowed is: %g' % (AdjustableProbabilityError, Template.ToleranceForInitialProbabilityOptimization)
        print ' Mean Probability Statistics reached are: ' + str(ResultProbabilitiesMeans)
        print ' STD Probability Statistics reached are: ' + str(ResultProbabilitiesSTDs)
        print ' Mean Odds Ratios Statistics reached are: ' + str(ResultOddsRatiosMeans)
        print ' STD Odds Ratios Statistics reached are: ' + str(ResultOddsRatiosSTDs)
        print ' Mean Division Ratios Statistics reached are: ' + str(ResultDivisionRatiosMeans)
        print ' STD Division Ratios Statistics reached are: ' + str(ResultDivisionRatiosSTDs)
        if Template.IterationStrategy is None:
            if (AdjustableProbabilityError < Template.ToleranceForInitialProbabilityOptimization):
                print 'Exiting loop since probability accuracy was achieved'
                break
            if (Template.IterationNumberForInitialProbabilityOptimization >= Template.MaxIterationsForInitialProbabilityOptimization):
                print 'Exiting loop due to sufficient number of iterations'
                break
        else:
            if (Template.IterationNumberForInitialProbabilityOptimization >= len( Template.IterationStrategy )):
                print 'Exiting loop since Iteration Strategy was complete'
                break
        # transfer probability result to new iteration
        CalculateNewProbability (Template, ResultProbabilitiesMeans)
            

    if DaskClient is not None:
        DaskClient.close()
    return (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs)


def Main(YamlFileName):
    "Run simulations using  instructions in Yaml file name"
    print (os.getcwd())    
    OptimizationParametersTemplate = OptimizationParametersClass()
    OptimizationParametersTemplate.LoadYamlData(YamlFileName)
    (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs) = LaunchDistributedSimulations(OptimizationParametersTemplate)
    return (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs)


def RunPaperResults():
    "Run all simulations for the paper - this demonstrates the two step solution"
    # Runs all simulations for the paper
    # "Population Disease Occurrence Models Using Evolutionary Computation"
    # The script will run both steps and if Holoviews is installed it will
    # also view visualization
    try:
        import CreatePlots
    except:
        print 'Warning: could not load CreatePlots.py - possibly since Holoviews is not installed.'
        print '         To generate plots install Holoviews and after simulation run the following command:'
        print '         python CreatePlots.py'
        
    # This function looks for these files to run step 1 and 2
    Step1Instructions = 'PopDOM1.yaml'
    Step2Instructions = 'PopDOM2.yaml'
    InstructionFileName = 'PopDOM.yaml'
    TextToReplaceInStep2 = '@@@@@@@@@@@@@@@@'
    StrategiesStart = ord('A')
    
    Step2InstructionsFile = open(Step2Instructions)
    InstructionsText2 = Step2InstructionsFile.read()
    Step2InstructionsFile.close()
    
    print 'Running Step 1'
    (ResultProbabilitiesMeans, ResultProbabilitiesSTDs, ResultOddsRatiosMeans, ResultOddsRatiosSTDs, ResultDivisionRatiosMeans, ResultDivisionRatiosSTDs) = Main(Step1Instructions)


    print 'Running Step 2'
    NumberOfStrategies = 2**len(ResultProbabilitiesMeans)
    for StrategyEnum in range(NumberOfStrategies):
        # construct simulations
        StragetyBinary = bin(NumberOfStrategies+StrategyEnum)[3:]
        StrategyName = chr(StrategiesStart+StrategyEnum)
        StrategyDivisionRatio = [  ResultDivisionComponent if StrategyBit=='1' else None  for (StrategyBit,ResultDivisionComponent) in zip(StragetyBinary,ResultDivisionRatiosMeans) ]
        NewInstructions = InstructionsText2.replace(TextToReplaceInStep2,repr(StrategyDivisionRatio))
        try:
            os.mkdir(StrategyName)
        except:
            print 'Bypassing creating the directory ' + StrategyName
        os.chdir(StrategyName)
        NewInstructionsFile = open(InstructionFileName,'w')
        NewInstructionsFile.write(NewInstructions)
        NewInstructionsFile.close()
        (ResultProbabilitiesMeans2, ResultProbabilitiesSTDs2, ResultOddsRatiosMeans2, ResultOddsRatiosSTDs2, ResultDivisionRatiosMeans2, ResultDivisionRatiosSTDs2) = Main(InstructionFileName)
        os.chdir('..')
    try:
        CreatePlots.CreatePlots()
    except:
        print 'Skipping final plot creation. to generate plots make sure Holoviews is installed'
        print 'and externally run this command:'
        print 'python CreatePlots.py'
        
 




OptimizationParametersTemplate = OptimizationParametersClass()

if __name__ == '__main__':
    print "USAGE: python PopDOM.py <InstructionsFile.yaml | PAPER>"
    print " If no argument provided the system will look by default for PopDOM.yaml "
    print " If PAPER is provided as argument, the program will run paper results"
    print " If another argument is provided, the system will look for instructions in this file"
    
    YamlFileName = 'PopDOM.yaml'
    if len(sys.argv) > 1:
        YamlFileName = sys.argv[1]
    if YamlFileName == 'PAPER':
        RunPaperResults()
    else:
        Main(YamlFileName)
