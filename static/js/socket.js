let socket = null;
let socket_connecting = false;
function connectSocket(callback){
    if (socket !== null){
        if (typeof callback === 'function') {
            callback();
        }
        return;
    }
    socket = io(host, {reconnectionAttempts: 3});
    // socket.connect(host);
    socket.on('connect_error', function(){
        socket = null;
        changeNumOfSOS(0);
    });
    socket.on('connect', ()=>{
        if (typeof callback === 'function') {
            callback();
        }
    });
    socket.on('disconnect', function(){
        socket = null;
        sos_poly = '';
        changeNumOfSOS(0);
    });

    socket.on('preprocess', function(data){
    });

    socket.on('rootangents', function(data){
        let str = data.rootsinfo;
        // /n -> <br>
        let i = 0;
        while (i < str.length){
            if (str[i] == '\n'){
                str = str.slice(0, i) + ' <br> ' + str.slice(i+1, str.length);
                i += 5;
            }
            ++i;
        }
        document.getElementById("rootsinfo").innerHTML = str;
        
        str = data.tangents;
        if (Array.isArray(str)){
            str = str.join('\n');
        }
        document.getElementById("input_tangents").value = str;
    });

    socket.on('sos', function(data){
        sos_results.latex = data.latex; 
        sos_results.txt   = data.txt;
        sos_results.formatted = data.formatted;
        document.getElementById('output_result').innerHTML = data[sos_results.show]

        const shadow_box = document.getElementById("shadow_box");
        let past;
        if ((past = shadow_box.firstChild)){
            past.remove();
        }
        
        let str;
        if (!data.success){
            str = '\\quad\\quad\\quad{\\rm Failed}\\quad\\quad\\quad';
        }else{
            const text_length = sos_results.latex.length;
            str = sos_results.latex.slice(2, text_length-2);
            if (str.indexOf('aligned') < 0){ // no aligned environment
                if (str.indexOf('\\\\') >= 0){
                    // 存在换行, 使用 \begin{aligned}
                    let i = 0;
                    while (i+1 < str.length){
                        if (str[i] == '\\' && str[i+1] == '\\'){
                            str = str.slice(0, i+2) + ' & ' + str.slice(i+2, str.length);
                            i += 2;        
                        }
                        ++i;
                    }
                    str = '\\begin{aligned}  \\  &' + str + '\\end{aligned}';
                }
            }
        }

        changeNumOfSOS(sos_work.num - 1);

        let svg = window.MathJax.tex2svg(str).children[0];
        // shadow_box.style.height = '80%';
        shadow_box.appendChild(svg);
        svg.setAttribute('style', 'width: 95%; height: 90%; margin-top: 2%'); 
        shadow_box.hidden = "";
        document.getElementById("shadow").hidden = "";
        document.getElementById("input_poly").blur();
        document.getElementById("input_tangents").blur();
        sos_results.displaying = 1;
    });
}