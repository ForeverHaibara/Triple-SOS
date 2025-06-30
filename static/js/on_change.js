var sos_work = {num: 0};
let constraints_locked = false;

function changeNumOfSOS(num){
    /* 
    A function that counts the number of SOSs in progress and changes the display of the number.
    */
    sos_work.num = Math.max(0, num);
    if (sos_work.num == 0){
        document.getElementById("sumofsquares").innerHTML = "Sum of Squares";
    }else{
        document.getElementById("sumofsquares").innerHTML = "Sum of Squares (" + sos_work.num + ")";
    }
}

function setPermChange(){
    /*
    Change the permutation group.
    */
    const x = document.querySelector('input[name="setting_perm"]:checked').value;
    const p = document.getElementById('setting_perm_input');
    const nvars = document.getElementById('setting_gens_input').value.length;
    if (x === 'cyc'){
        p.disabled = true;
        p.value = `[[${Array.from({length: nvars}, (_, i) => (i+1) % nvars)}]]`;
    }else if (x === 'sym'){
        p.disabled = true;
        let array1 = Array.from({length: nvars}, (_, i) => (i+1) % nvars);
        if (nvars <= 2){ p.value = `[[${array1}]]`; return; }
        let array2 = Array.from({length: nvars}, (_, i) => i % nvars);
        array2[0] = 1;
        array2[1] = 0;
        p.value = `[[${array1}],[${array2}]]`;
    }else if (x === 'custom'){
        p.disabled = false;
    }
}

function setGeneratorsChange(event) {
    /*
    Change the generators of the polynomial ring.
    */
    const input = event.target;
    const currentValue = input.value;

    let validString = '';
    function isValidChar(char) {
        const regex = /^[a-z]$/;
        return regex.test(char) && !['s', 'p'].includes(char) && !validString.includes(char);
    }
    for (let i = 0; i < currentValue.length; i++) {
        const char = currentValue[i];
        if (isValidChar(char)){
            validString += char;
        }
    }
    input.value = validString;
    setPermChange();

    if (constraints_locked){
        const tableBody = document.getElementById('constraints_table_body');
        tableBody.innerHTML = '';
        const gensInput = validString;
        const chars = gensInput.split('');
        chars.forEach(char => {
            if (char.trim()) constraintsAddRow('≥0', disable=true, constraint=char);
        });
    }
}

document.getElementById('setting_gens_input').addEventListener('input', setGeneratorsChange);


/************************************************************
* 
*                      Constraints Area
*
*************************************************************/
const constraints_constraint_style = "font-size: 10px; padding: 0.1rem 0.3rem; height: 20px;";
const constraints_trash_svg = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" class="bi bi-trash" viewBox="0 0 16 16">
    <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
    <path fill-rule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
</svg>`;
const constraints_type_obj = ((type) => `<td class="text-center constraints-type" align="center" valign="middle" style="padding: 0.25rem 0.5rem;">${type}</td>`);

function constraintsAddRow(type, locked=false, constraint='', alias='') {
    if ((!locked)&&constraints_locked) constraintsToggleLock();
    const tableBody = document.getElementById('constraints_table_body');
    const newRow = document.createElement('tr');
    if (!locked){
        newRow.innerHTML = `
            <td><input type="text" class="form-control form-control-xs constraints-constraint" style="${constraints_constraint_style}" value="${constraint}" placeholder="e.g., a+b+c-1"></td>
            <td><input type="text" class="form-control form-control-xs constraints-alias" style="${constraints_constraint_style}" value="${alias}" placeholder="e.g., f(a,b,c)"></td>
            ${constraints_type_obj(type)}
            <td class="text-center" style="padding: 0.25rem 0.5rem;">
                <button class="btn btn-xs btn-danger" style="padding: 0.1rem 0.3rem; font-size: 10px; height: 20px;" onclick="this.closest('tr').remove()">
                    ${constraints_trash_svg}
                </button>
            </td>
        `;
    }else{
        newRow.innerHTML = `
            <td><input type="text" class="form-control form-control-xs constraints-constraint bg-transparent border-0 shadow-none" style="${constraints_constraint_style} border: none; box-shadow: none; background: transparent;" value="${constraint}" readonly></td>
            <td><input type="text" class="form-control form-control-xs constraints-alias bg-transparent border-0 shadow-none" style="${constraints_constraint_style} border: none; box-shadow: none; background: transparent;" value="${alias}" placeholder="" readonly></td>
            ${constraints_type_obj(type)}
            <td class="text-center" style="padding: 0.25rem 0.5rem;">
                <button class="btn btn-xs btn-danger" style="padding: 0.1rem 0.3rem; font-size: 10px; height: 20px;" onclick="this.closest('tr').remove()" disabled>
                    ${constraints_trash_svg}
                </button>
            </td>
        `;
    }
    tableBody.appendChild(newRow);
}

function constraintsAddPositiveGens(clear=false) {
    const tableBody = document.getElementById('constraints_table_body');
    if (clear)
        tableBody.innerHTML = '';
    const gensInput = document.getElementById('setting_gens_input').value;
    const chars = gensInput.split('');
    chars.forEach(char => {
        if (char.trim()) {
            constraintsAddRow('≥0', disable=true, constraint=char);
        }
    });
}

function constraintsClear() {
    if (constraints_locked) constraintsToggleLock();
    document.getElementById('constraints_table_body').innerHTML = '';
}
function constraintsToggleLock() {
    constraints_locked = !constraints_locked;
    const lockButton = document.getElementById('constraints-lock');
    const lockIcon = document.getElementById('lockIcon');
    const tableBody = document.getElementById('constraints_table_body');
    // const table = tableBody.closest('table');
    
    if (constraints_locked) {
        // switch to locked
        lockIcon.classList.remove('bi-unlock');
        lockIcon.classList.add('bi-lock');
        lockButton.classList.remove('btn-outline-success');
        lockButton.classList.add('btn-success');
        // table.classList.add('table-locked');
        
        // clear the table and insert gens >= 0
        constraintsAddPositiveGens(clear=true);
    } else {
        // switch to unlocked
        lockIcon.classList.remove('bi-lock');
        lockIcon.classList.add('bi-unlock');
        lockButton.classList.remove('btn-success');
        lockButton.classList.add('btn-outline-success');
        // table.classList.remove('table-locked');
        
        // 解锁时启用输入和删除按钮
        const rows = tableBody.querySelectorAll('tr');
        rows.forEach(row => {
            const inputs = row.querySelectorAll('input');
            inputs.forEach(input => {
                    input.removeAttribute('readonly');
                    input.classList.remove('bg-transparent', 'border-0', 'shadow-none');
                    input.style.border = '';
                    input.style.boxShadow = '';
                    input.style.background = '';
                });
            const deleteBtn = row.querySelector('button');
            if (deleteBtn) deleteBtn.removeAttribute('disabled');
        });
    }
}
constraintsToggleLock(); // initialization

function constraintsAddTools(name, type){
    const gens = document.getElementById('setting_gens_input').value.split('');
    const n = gens.length;
    function getConstraints(){
        if (type === '≥0'){
            if (name === 'triangle'){
                if (n > 2){
                    return gens.map((_, i) => {
                        const otherElements = [];
                        for (let j = 1; j < n; j++) {
                            otherElements.push(gens[(i + j) % n]);
                        }
                        const sumPart = otherElements.join('+');
                        return `${sumPart}-${gens[i]}`;
                    });
                }
            }else if (name === 'acute triangle'){
                if (n > 2){
                    return gens.map((_, i) => {
                        const otherElements = [];
                        for (let j = 1; j < n; j++) {
                            otherElements.push(gens[(i + j) % n] + '^2');
                        }
                        const sumPart = otherElements.join('+');
                        return `${sumPart}-${gens[i]}^2`;
                    });
                }
            }
        }
        return [];
    }
    getConstraints().forEach((constraint) => {
        constraintsAddRow(type, locked=false, constraint=constraint);
    });
}





document.addEventListener('click', function(event) {
    // 检查点击事件是否发生在弹出框之外
    if (sos_results.displaying == 1 && !event.target.closest('#shadow_box') && 
            !event.target.closest('#output_showlatex')){
        // 如果点击在弹出框之外，则关闭弹出框
        sos_results.displaying = 0; 
        // document.getElementById("shadow_box").hidden = "hidden";
        document.getElementById("shadow").hidden = "hidden";
    }
});

document.onkeyup = function(e) {
    // 兼容FF和IE和Opera
    var event = e || window.event;
    var key = event.which || event.keyCode || event.charCode;
    if ((key == 13 || key == 27) && sos_results.displaying == 1){
        // is displaying
        sos_results.displaying = 0; 
        // document.getElementById("shadow_box").hidden = "hidden";
        document.getElementById("shadow").hidden = "hidden";
    }
};

function changeShowType(x){
    if (sos_results.show != x){
        document.getElementById('output_show' + sos_results.show).style.color = 'black';
        document.getElementById('output_show' + sos_results.show).style.border = 'none';
        sos_results.show = x;
        document.getElementById('output_show' + x).style.color = 'rgb(41,50,225)';
        document.getElementById('output_show' + x).style.borderBottom = '2px solid blue';
        document.getElementById('output_result').innerHTML = sos_results[x];
    }else if (x == 'latex'){ // == sos_results.show
        document.getElementById("shadow_box").hidden = "";
        document.getElementById("shadow").hidden = "";
        sos_results.displaying = 1;
    }
}

// function changeShowTypeHover(x, event){
//     // event == 0: mouseenter    event == 1: mouseleave
//     if (sos_results.show != x){
//         if (event == 0){
//             document.getElementById('output_show' + x).style.color = 'rgb(19,135,245)';
//         }else{
//             document.getElementById('output_show' + x).style.color = 'black';
//         }
//     }
// }

function initDropDownMenu(){
    const dropdown_items = document.querySelectorAll(".dropdown_item");
    let dropdown_display = [];
    dropdown_items.forEach((item, i) => {
        const submenu = item.querySelector(".submenu");
        dropdown_display.push("none");
        // item.addEventListener("click", (event) => {
        //     event.stopPropagation();
        //     if (!submenu.contains(event.target)){
        //         // click on the button, not the submenu
        //         dropdown_display[i] = dropdown_display[i] == "block" ? "none" : "block";
        //         submenu.style.display = dropdown_display[i];

        //         // also close other dropdown menus
        //         dropdown_items.forEach((item, j) => {
        //             if (j != i && dropdown_display[j] == "block") {
        //                 dropdown_display[j] = "none";
        //                 item.querySelector(".submenu").style.display = "none";
        //             }
        //         });
        //     }
        // });
        item.addEventListener("mouseenter", () => {
            if (dropdown_display[i] == "none") {
                dropdown_display[i] = "block";
                submenu.style.display = "block";
            }
        });
        item.addEventListener("mouseleave", () => {
            // settimeout check: if the mouse is still in the submenu, do not close the submenu
            setTimeout(() => {
                if (dropdown_display[i] == "block" && !submenu.matches(":hover")) {
                    dropdown_display[i] = "none";
                    submenu.style.display = "none";
                }
            }, 100);
        });
    });
    // clicking elsewhere will close all dropdown menus (note that cases of clicking on submenus are blocked by stopPropagation)
    // document.addEventListener("click", () => {
    //     dropdown_items.forEach((item, i) => {
    //         if (dropdown_display[i] == "block") {
    //             dropdown_display[i] = "none";
    //             item.querySelector(".submenu").style.display = "none";
    //         }
    //     });
    // });
}
initDropDownMenu();