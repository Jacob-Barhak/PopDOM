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
This script generates the plots for the MODSIM 2019 paper 
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
import numpy
import PopDOM
import glob
import pandas
import os
import webbrowser
import holoviews 
import operator

import holoviews.plotting.bokeh


def CreatePlots():
    "Generate the plots for the paper"
    YamlFilePattern = '*'+os.sep+'PopDOM.yaml'
    YamlFileNames = sorted(glob.glob(YamlFilePattern))
    OptimizationParametersTemplateList = []

    MainDataFrame = None

    for (EnumResultSet,YamlFileName) in enumerate(YamlFileNames):
        Directory = YamlFileName.split(os.sep)[0]
        OptimizationParametersTemplate = PopDOM.OptimizationParametersClass()
        OptimizationParametersTemplate.LoadYamlData(YamlFileName)
        RequestedProbabilities = OptimizationParametersTemplate.Probabilities
        RequestedOddsRatios = OptimizationParametersTemplate.OddsRatios
        RequestedDivisionRatios = OptimizationParametersTemplate.DivisionRatios
        OptimizationParametersTemplateList.append(OptimizationParametersTemplate)
        OptimizationParametersTemplateList.append(OptimizationParametersTemplate)
        AggregateOutputFileName = OptimizationParametersTemplate.AggregateOutputFileName
        FileNamePattern = Directory + os.sep + AggregateOutputFileName
        ProbColumnNames = ['Prob '+ str(Entry) for Entry in OptimizationParametersTemplate.ProbabilitiesTitlesResolved]
        OddsColumnNames = ['Odds '+ str(Entry) for Entry in OptimizationParametersTemplate.OddsRatiosTitlesResolved]
        DivColumnNames = ['Div '+ str(Entry) for Entry in OptimizationParametersTemplate.OddsRatiosTitlesResolved]
        
        RefProbColumnNames = ['Ref Prob ' + str(Entry) for Entry in OptimizationParametersTemplate.ProbabilitiesTitlesResolved]
        RefOddsColumnNames = ['Ref Odds ' + str(Entry) for Entry in OptimizationParametersTemplate.OddsRatiosTitlesResolved]
        RefDivColumnNames = ['Ref Div ' + str(Entry) for Entry in OptimizationParametersTemplate.OddsRatiosTitlesResolved]
        
        
        HeaderToUseForLoad = ['Dummy0', 'Repetition', 'Dummy1'] + ProbColumnNames + ['Dummy2'] + OddsColumnNames + ['Dummy3'] + DivColumnNames
        HeaderToKeepAfterLoad = ['Repetition'] + ProbColumnNames + OddsColumnNames + DivColumnNames
        
        if MainDataFrame is None:
            AddedHeaders = ['Treatment Level', 'Strategy'] + RefProbColumnNames + RefOddsColumnNames + RefDivColumnNames
            DataHeaders = HeaderToKeepAfterLoad + AddedHeaders
            MainDataFrame = pandas.DataFrame(columns = DataHeaders)
        StepEnum = 0
        while (True):
            FileNameToLoad = FileNamePattern%StepEnum
            try:
                DataFrameRaw = pandas.read_csv(FileNameToLoad, header=None, names = HeaderToUseForLoad, usecols = HeaderToKeepAfterLoad )
            except:
                break

            AddedDataList = [StepEnum, Directory[-1]] + [ abs(Entry) if Entry is not None else None for Entry in (RequestedProbabilities + RequestedOddsRatios + RequestedDivisionRatios) ]
            for (AddedHeader,AddedData) in zip(AddedHeaders,AddedDataList):
                DataFrameRaw.loc[:,AddedHeader] = AddedData
            MainDataFrame = pandas.concat([MainDataFrame, DataFrameRaw])
            StepEnum = StepEnum + 1
        MaxRepetition = StepEnum-1
            

    print MainDataFrame.columns

    holoviews.Store.current_backend = 'bokeh'

    HoloviewsDataSet = holoviews.Dataset(MainDataFrame, ['Strategy', 'Repetition', 'Treatment Level'], ProbColumnNames + OddsColumnNames  + DivColumnNames)
    HoloviewsAggregate = HoloviewsDataSet.aggregate( ['Strategy', 'Treatment Level'], function=numpy.mean, spreadfn=numpy.std)

    print 'Base'
    print HoloviewsDataSet
    print 'Aggregate'
    print HoloviewsAggregate
    
    AggregateDataFrame = HoloviewsAggregate.dframe()
    AggregateDataFrame.to_csv('AllStats.csv')
    AggregateDataFrame.loc[AggregateDataFrame['Treatment Level']==MaxRepetition].to_csv('FinalStats.csv')

    
    ColorList = ['red', 'green', 'blue','magenta','cyan','pink','brown','yellow']
    MarkerList = ['x', '^', '+','s']

    def CreatePlot(ResultsNames, VerticalAxisLabel):
        "Create overlay plot of column names"
        def MyCycle(List, Enum):
            "Return a value from the list"
            RetVal = List[Enum%len(List)]
            return RetVal
        PlotList = [(holoviews.GridSpace(HoloviewsDataSet.to(holoviews.Scatter,['Treatment Level'],PlotOutcome,  label = VerticalAxisLabel, group = PlotOutcome).redim.label(**{PlotOutcome:VerticalAxisLabel}).overlay(['Repetition']).options({'Scatter': { 'color':MyCycle(ColorList,Enum), 'marker':MyCycle(MarkerList,Enum), 'size': 2, 'legend_position':'right', 'show_legend':True} , 'NdOverlay': {'legend_position':'right', 'show_legend':True}, 'Overlay': {'legend_position':'right', 'show_legend':True}} )).options({'GridSpace': { 'shared_yaxis':True, 'shared_xaxis':True}}) ) for (Enum,PlotOutcome) in enumerate(ResultsNames) ]
        PlotListAggregate = [(holoviews.GridSpace(HoloviewsAggregate.to(holoviews.Curve,['Treatment Level'],PlotOutcome,  label = VerticalAxisLabel, group = PlotOutcome).redim.label(**{PlotOutcome:VerticalAxisLabel}).options({'Curve': { 'color':MyCycle(ColorList,Enum), 'show_legend':True} , 'NdOverlay': {'legend_position':'right', 'show_legend':True}} )).options({'GridSpace': { 'shared_yaxis':True, 'shared_xaxis':True}}) ) for (Enum,PlotOutcome) in enumerate(ResultsNames) ]
        HoloviewsText = [ holoviews.Text(0.5,(Enum+1)/(len(ResultsNames)+1)*0.7,PlotOutcome).options({'Text': { 'color':MyCycle(ColorList,Enum), 'xaxis': None, 'yaxis': None, 'show_frame': False}}) for (Enum,PlotOutcome) in enumerate(ResultsNames) ]


        CreatedPlot = reduce (operator.mul, PlotList)
        CreatePlotAggregate = reduce (operator.mul, PlotListAggregate)
        CreatePlotTexts = reduce (operator.mul, HoloviewsText)
        
        return CreatedPlot, CreatePlotAggregate, CreatePlotTexts

    Plot1,Agg1,Txt1 = CreatePlot( ProbColumnNames, 'Probabilities')
    Plot2,Agg2,Txt2 = CreatePlot( OddsColumnNames, 'Odds Ratios')
    Plot3,Agg3,Txt3 = CreatePlot( DivColumnNames, ' Division Ratios')
    CombposedPlot = ((Plot1*Agg1+Txt1) + (Plot2*Agg2)+Txt2 + (Plot3*Agg3)+Txt3)
    
    Plot = CombposedPlot.cols(2)
    print "Composed"
    print CombposedPlot

    Renderer = holoviews.renderer('bokeh')

    Renderer.save(Plot, 'HoloviewsPlot')
    webbrowser.open('HoloviewsPlot.html') 


if __name__ == '__main__':
    CreatePlots()
