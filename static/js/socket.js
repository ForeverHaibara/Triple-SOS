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
        
        // str = data.tangents;
        // if (Array.isArray(str)){
        //     str = str.join('\n');
        // }
        // document.getElementById("input_tangents").value = str;
    });

    socket.on('sos', function(data){
        changeNumOfSOS(sos_work.num - 1);
        setSOSResult(data);

        // record the result to the history
        setHistoryByTimestamp(data.timestamp, 'sos_results', data);

        document.getElementById("shadow").hidden = "";
        document.getElementById("input_poly").blur();
        // document.getElementById("input_tangents").blur();
        sos_results.displaying = 1;
    });
}