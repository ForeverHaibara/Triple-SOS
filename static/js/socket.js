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

    socket.on('findroots', function(data){
        const roots = data.rootsinfo;
        const trunc = roots.slice(0, Math.min(5, roots.length));
        let roots_string = 'Local Minima Approx:<br>' +
            trunc.map(root => "(" + root.join(", ") + ")").join("<br>");
        document.getElementById("rootsinfo").innerHTML = roots_string;
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