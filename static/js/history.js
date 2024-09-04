let history_data = [];
const displayHistoryMaxChars = 23;

function updateHistoryData() {
    const historyDiv = document.getElementById('history_panel');
    // delete all history records
    while (historyDiv.firstChild) {
        historyDiv.removeChild(historyDiv.firstChild);
    }
    history_data.forEach((item, index) => {
        const record = document.createElement('div');
        function displayPoly(poly) {
            if (poly.length <= displayHistoryMaxChars) {
                return poly;
            }
            return poly.slice(0, displayHistoryMaxChars/2-1) + ' ··· ' + poly.slice(1-displayHistoryMaxChars/2);
            // return poly.slice(0, displayHistoryMaxChars - 2) + ' ...';
        }
        record.innerHTML = `${displayPoly(item.poly)}`;

        record.style = 'cursor: pointer; margin: 0; padding: 0px; border: none; font-size: 13px; background-color: ' + 
            (index % 2 === 1 ? 'white' : 'rgb(245,245,245)') + ';';
        
        // 添加hover效果
        record.addEventListener('mouseover', () => {
            // record.classList.add('bg-light');
            record.style.backgroundColor = '#afeeee';
        });
        record.addEventListener('mouseout', () => {
            // record.classList.remove('bg-light');
            record.style.backgroundColor = (index % 2 === 1 ? 'white' : 'rgb(245,245,245)');    
        });

        // 点击事件
        record.addEventListener('click', () => {
            document.getElementById('input_poly').value = item.poly;
            document.getElementById('setting_gens_input').value = item.gens;
            document.getElementById('setting_perm_input').value = item.perm;
            document.querySelector('input[name="setting_perm"][value="' + item.perm_choice + '"]').checked = true;
            setPermChange();
        });

        historyDiv.appendChild(record);
    });
}
updateHistoryData();