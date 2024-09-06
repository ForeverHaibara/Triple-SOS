let history_data = [];
let _history_timestamp_to_ind = {};
const displayHistoryMaxChars = 600;
const COLOR_GRAY = 'rgb(230,230,230)';
const COLOR_SOS_SUCCESS = '#1d6300';

function setHistoryByTimestamp(timestamp, key, value){
    /*
    Set the history data by timestamp.
    */
    const ind = _history_timestamp_to_ind[timestamp];
    if (ind !== undefined){
        history_data[ind][key] = value;
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
    const keys = ['poly', 'gens', 'perm', 'perm_choice'];
    for (let key of keys){
        if (history_data[0][key] !== data[key]){
            return false;
        }
    }
    return true;
}


function _setRecordProperties(record, index){
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
        document.getElementById('setting_gens_input').value = data.gens;
        document.getElementById('setting_perm_input').value = data.perm;
        document.querySelector('input[name="setting_perm"][value="' + data.perm_choice + '"]').checked = true;
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
        function displayPoly(poly) {
            if (poly.length <= displayHistoryMaxChars) {
                return poly;
            }
            return poly.slice(0, displayHistoryMaxChars/2-1) + ' ··· ' + poly.slice(1-displayHistoryMaxChars/2);
            // return poly.slice(0, displayHistoryMaxChars - 2) + ' ...';
        }
        record.innerHTML = `${displayPoly(item.poly)}`;

        _setRecordProperties(record, index);

        historyDiv.appendChild(record);
    });
}
updateHistoryData();