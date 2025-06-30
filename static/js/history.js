let history_data = [];
let _history_timestamp_to_ind = {};
const displayHistoryMaxChars = 600;
const COLOR_GRAY = 'rgb(230,230,230)';
const COLOR_SOS_SUCCESS = '#1d6300';

function _historyDisplayPoly(poly) {
    if (poly.length <= displayHistoryMaxChars) {
        return poly;
    }
    return poly.slice(0, displayHistoryMaxChars/2-1) + ' ··· ' + poly.slice(1-displayHistoryMaxChars/2);
    // return poly.slice(0, displayHistoryMaxChars - 2) + ' ...';
}

function setHistoryByTimestamp(timestamp, key, value){
    /*
    Set the history data by timestamp.
    */
    const ind = _history_timestamp_to_ind[timestamp];
    if (ind !== undefined){
        history_data[ind][key] = value;
    }
    if (key === 'vis' && value.txt !== undefined){
        history_data[ind].poly = value.txt
        history_data[ind].div.innerHTML = _historyDisplayPoly(value.txt);
    }
    return ind;
}

function isSameHistoryRecord(data){
    /*
    Check whether the given data is the same as the leading history record,
    excluding the timestamp and the results.
    */
    if (history_data.length === 0){
        return false;
    }
    function isEqual(a, b) {
        if (typeof a === 'string' && typeof b === 'string') {
            return a.trim() === b.trim();
        }

        if (typeof a === 'object' && a !== null && typeof b === 'object' && b !== null) {
            const keysA = Object.keys(a);
            const keysB = Object.keys(b);

            if (keysA.length !== keysB.length) return false;

            for (const key of keysA) {
                if (!keysB.includes(key) || !isEqual(a[key], b[key])) {
                    return false;
                }
            }
            return true;
        }

        return a === b;
    }

    const keys = ['poly', 'gens', 'perm', 'perm_choice', 'ineq_constraints', 'eq_constraints'];
    for (let key of keys){
        if (!isEqual(history_data[0][key], data[key])) {
            return false;
        }
    }
    return true;
}


function _setRecordProperties(record, index){
    /*
    Fill in the forms from history records.
    */
    record._history_index = index;
    const data = history_data[index];
    record.style = 'cursor: pointer; margin: 0; padding: 0px; border: none; font-size: 13px; background-color: ' + 
        (index % 2 === 1 ? 'white' : `${COLOR_GRAY}`) + ';' +
        `color: ${((data.sos_results!==undefined)&&(data.sos_results.success)) ? COLOR_SOS_SUCCESS : 'black'};`;

    record.addEventListener('mouseover', () => {
        record.style.backgroundColor = '#afeeee';
    });
    record.addEventListener('mouseout', () => {
        record.style.backgroundColor = (record._history_index % 2 === 1 ? 'white' : `${COLOR_GRAY}`);    
    });
    record.addEventListener('click', () => {
        document.getElementById('input_poly').value = data.poly;

        // set generators and permutation group
        document.getElementById('setting_gens_input').value = data.gens;
        document.getElementById('setting_perm_input').value = data.perm;
        document.querySelector('input[name="setting_perm"][value="' + data.perm_choice + '"]').checked = true;

        // judge whether the constraints are gens >= 0
        function _constraintShouldLock(){
            const gens = data.gens;//.split('');
            const n = gens.length;
            if (Object.keys(data.eq_constraints).length === 0 && Object.keys(data.ineq_constraints).length === n){
                for (let i=0;i<n;++i){
                    if (data.ineq_constraints[gens[i]] !== '') return false;
                }
                return true;
            }
            return false;
        }
        const constraint_should_lock = _constraintShouldLock();

        // set constraints
        const constraint_table = document.getElementById('constraints_table_body');
        if (constraints_locked !== constraint_should_lock){
            constraintsToggleLock();
        }
        if (constraints_locked){
            constraintsAddPositiveGens(clear=true);
        }
        if (!constraint_should_lock){
            constraint_table.innerHTML = '';
            Object.entries(data.ineq_constraints).forEach(
                (x) => constraintsAddRow('≥0', constraint_should_lock, constraint=x[0], alias=x[1]))
            Object.entries(data.eq_constraints).forEach(
                (x) => constraintsAddRow('=0', constraint_should_lock, constraint=x[0], alias=x[1]))
        }

        renderVisualization(data.vis);
        setSOSResult(data.sos_results);
        setPermChange();
    });
}

function updateHistoryData() {
    /*
    When the list of history_data is updated, update the history panel and other related display.
    */
    const historyDiv = document.getElementById('history_panel');
    // delete all history records
    while (historyDiv.firstChild) {
        historyDiv.removeChild(historyDiv.firstChild);
    }
    _history_timestamp_to_ind = {};

    if (history_data.length === 0) {
        // display an empty history panel with a gray strip
        const record = document.createElement('div');
        record.innerHTML = '&nbsp;';
        record.style = `cursor: pointer; margin: 0; padding: 0px; border: none; font-size: 13px; background-color: ${COLOR_GRAY};`;
        historyDiv.appendChild(record);
        return;
    }

    history_data.forEach((item, index) => {
        const record = document.createElement('div');
        history_data[index].div = record;
        _history_timestamp_to_ind[item.timestamp] = index;
        record.innerHTML = `${_historyDisplayPoly(item.poly)}`;

        _setRecordProperties(record, index);

        historyDiv.appendChild(record);
    });
}
updateHistoryData();