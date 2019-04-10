using Optim, CSV, DataFrames, Random

read(file) = CSV.read("$file.csv")

games, optionCounts = begin
    games = Set()
    optionCounts = Dict()
    for row in eachrow(read("Option Counts"))
        game = Symbol(row.Game)
        push!(games, game)
        optionCounts[game, "R"] = row.R
        optionCounts[game, "C"] = row.C
    end
    games, optionCounts
end

types, typeMoves, gameMoves = begin
    input = read("Types")
    types = []
    typeMoves = Dict()
    gameMoves = Dict()
    for type in eachrow(input)
        role = type.Role
        if role == "R"
            push!(types, Symbol(type.Type))
        end
        for game in games
            typeMoves[Symbol(type.Type), role, game] = type[game]
            if !haskey(gameMoves, (game, role))
                gameMoves[game, role] = Set()
            end
            push!(gameMoves[game, role], type[game])
        end
    end
    types, typeMoves, gameMoves
end

function denoteParams(params)
    if length(params) != 2 * length(types)
        throw(ArgumentError("wrong number of parameters"))
    end
    priors = Real[]
    errors = Real[]
    for i = 1:length(types)
        prior = params[2 * i - 1]
        error = params[2 * i]
        if prior < 0 || error < 0 || error > 1
            throw(ArgumentError("invalid parameter #$i"))
        end
        push!(priors, prior)
        push!(errors, error)
    end
    priors = priors / sum(priors) # normalize to total probability 1
    priors, errors
end

function llSubjectType(subject, type, error)
    role = subject[:Role]
    ll = 0
    for game in games
        optionCount = optionCounts[game, role]
        ll += log(subject[game] == typeMoves[type, role, game]
                  ? 1 - error * (optionCount - 1) / optionCount
                  : error / optionCount)
    end
    ll
end

nll(subjects) = function(params)
    local priors, errors
    try
        priors, errors = denoteParams(params)
    catch
        return Inf
    end
    logLikelihood = 0
    for subject in eachrow(subjects)
        subjectLikelihood = 0
        for i = 1:length(types)
            type = types[i]
            prior = priors[i]
            error = errors[i]
            subjectLikelihood += prior * exp(llSubjectType(subject, type, error))
        end
        logLikelihood += log(subjectLikelihood)
    end
    -logLikelihood
end

msd(subjects) = function(params)
    local priors, errors
    try
        priors, errors = denoteParams(params)
    catch
        return Inf
    end
    totalSquaredDeviation = 0
    n = 0
    for game in games
        for role in ["R", "C"]
            optionCount = optionCounts[game, role]
            relevantSubjects = filter(s -> s.Role == role, subjects)
            for move in gameMoves[game, role]
                expectedProportion = 0
                for i = 1:length(types)
                    type = types[i]
                    prior = priors[i]
                    error = errors[i]
                    if typeMoves[type, role, game] == move
                        expectedProportion += prior * (1 - error * (optionCount - 1) / optionCount)
                    else
                        expectedProportion += prior * error / optionCount
                    end
                end
                playedMove = filter(s -> s[game] == move, relevantSubjects)
                actualProportion = size(playedMove, 1) / size(relevantSubjects, 1)
                totalSquaredDeviation += (actualProportion - expectedProportion) ^ 2
                n += 1
            end
        end
    end
    totalSquaredDeviation / n
end

function showParams(params)
    priors, errors = denoteParams(params)
    DataFrame(Type = types, Prior = priors, Error = errors)
end

function doOptimization(f, initParams, iterations = 1, verbose = true)
    if verbose
        println(f(initParams), ", ", initParams)
    end
    function recur(params, iterations)
        res = optimize(f, params, iterations = iterations)
        if Optim.converged(res)
            res
        else
            v = Optim.minimum(res)
            params = Optim.minimizer(res)
            if verbose
                println(v, ", ", params)
            end
            recur(params, 2 * iterations)
        end
    end
    res = recur(initParams, iterations)
    v = Optim.minimum(res)
    params = Optim.minimizer(res)
    v, params, showParams(params)
end

doMLE(subjects, initParams, iterations = 1, verbose = true) =
  doOptimization(nll(subjects), initParams, iterations, verbose)

optimizeMSD(subjects, initParams, iterations = 1, verbose = true) =
  doOptimization(msd(subjects), initParams, iterations, verbose)

nullParams = [
  0.125, 1,
  0.125, 1,
  0.125, 1,
  0.125, 1,
  0.125, 1,
  0.125, 1,
  0.125, 1,
  0.125, 1
]

function crossValidate(fOpt, fEval, subjects, initParams, numFolds = 2)
    n = size(subjects, 1)
    foldAssignments = shuffle([1 + i % numFolds for i in 1:n])
    totalTrainingImprovement = 0
    totalTestImprovement = 0
    for foldNum in 1:numFolds
        inFold = [foldAssignments[i] == foldNum for i in 1:length(foldAssignments)]
        fold = subjects[inFold, :]
        rest = subjects[map(!, inFold), :]
        _, params, _ = doOptimization(fOpt(rest), initParams, 1, false)
        results = DataFrame(Dataset       = ["Training"              , "Test"                ],
                            NullParams    = [-fEval(rest)(nullParams), -fEval(fold)(nullParams)],
                            TrainedParams = [-fEval(rest)(params)    , -fEval(fold)(params)    ])
        improvement(row) = (results[row, :NullParams] - results[row, :TrainedParams]) / results[row, :NullParams]
        trainingImprovement = improvement(1)
        testImprovement = improvement(2)
        totalTrainingImprovement += trainingImprovement
        totalTestImprovement += testImprovement
        println("Fold ", foldNum, ": ", (trainingImprovement, testImprovement))
    end
    totalTrainingImprovement/numFolds, totalTestImprovement/numFolds
end

myInitParams = [
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5,
    0.125, 0.5
]

cgcbOpenBoxParams = [
    0.000, 1.000,
    0.000, 1.000,
    0.199, 0.285,
    0.344, 0.233,
    0.298, 0.276,
    0.000, 1.000,
    0.160, 0.165,
    0.000, 1.000
]

cgcbBaselineParams = [
    0.044, 0.253,
    0.000, 1.000,
    0.240, 0.286,
    0.496, 0.203,
    0.175, 0.704,
    0.000, 1.000,
    0.044, 0.163,
    0.000, 1.000
]

perfectNashParams = [
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    0, 0,
    1, 0,
    0, 0
]

subjects = read("Subjects")

obSubjects = filter(subject -> subject[Symbol("Treatment and Run")] == "OB", subjects)

baselineSubjects = filter(subject -> subject[Symbol("Treatment and Run")] in ["B1", "B2"], subjects)

doMLE # don't print a dataframe when the file is included from REPL
