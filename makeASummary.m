%% summary making helper
function arr = makeASummary(allResults)
    rowNames = string(allResults{1}.Row);
    arr = zeros(length(rowNames),length(rowNames));
    for i = 1:length(allResults)
        arr = arr + allResults{i}.Variables;
    end
end