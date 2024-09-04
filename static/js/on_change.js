var sos_work = {num: 0};

function changeNumOfSOS(num){
    /* 
    A function that counts the number of SOSs in progress and changes the display of the number.
    */
    sos_work.num = Math.max(0, num);
    if (sos_work.num == 0){
        document.getElementById("sumofsquare").innerHTML = "Sum of Square";
    }else{
        document.getElementById("sumofsquare").innerHTML = "Sum of Square (" + sos_work.num + ")";
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
}

document.getElementById('setting_gens_input').addEventListener('input', setGeneratorsChange);




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

function changeShowTypeHover(x, event){
    // event == 0: mouseenter    event == 1: mouseleave
    if (sos_results.show != x){
        if (event == 0){
            document.getElementById('output_show' + x).style.color = 'rgb(19,135,245)';
        }else{
            document.getElementById('output_show' + x).style.color = 'black';
        }
    }
}