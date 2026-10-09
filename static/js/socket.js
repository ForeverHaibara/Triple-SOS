let socket = null;
let socket_callbacks = [];
function connectSocket(callback, onError = showBackendPrompt){
    if (socket !== null && socket.connected){
        if (typeof callback === 'function') {
            callback();
        }
        return;
    }
    socket_callbacks.push({callback: callback, onError: onError});
    if (socket !== null) return;
    setBackendStatus('connecting');
    // Keep one connection attempt and discard its callbacks after the first result.
    socket = io(host, {reconnection: false, timeout: 4000, forceNew: true});
    const connecting_socket = socket;
    function discardSocket(){
        connecting_socket.removeAllListeners();
        connecting_socket.disconnect();
        socket = null;
    }
    socket.on('connect_error', function(){
        discardSocket();
        changeNumOfSOS(0);
        setBackendStatus('offline');
        const callbacks = socket_callbacks;
        socket_callbacks = [];
        callbacks.forEach(item => item.onError());
    });
    socket.on('connect', ()=>{
        setBackendStatus('connected');
        const callbacks = socket_callbacks;
        socket_callbacks = [];
        callbacks.forEach(item => {
            if (typeof item.callback === 'function') item.callback();
        });
    });
    socket.on('disconnect', function(){
        discardSocket();
        sos_poly = '';
        setBackendStatus('offline');
        if (sos_work.num > 0) showBackendPrompt();
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
