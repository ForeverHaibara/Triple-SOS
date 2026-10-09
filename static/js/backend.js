const backend_modal_element = document.getElementById('backend_modal');
const backend_modal = new bootstrap.Modal(backend_modal_element);
const backend_status = document.getElementById('backend_status');
const backend_message = document.getElementById('backend_message');
let backend_poll = null;
let backend_prompt_disabled = false;

try {
    backend_prompt_disabled = sessionStorage.getItem('triples_backend_prompt_disabled') === 'true';
} catch (error) {
    // File URLs may not have access to session storage. Keep an in-memory fallback.
}

function isLocalBackend(){
    return ['localhost', '127.0.0.1', '[::1]'].includes(new URL(host).hostname);
}

function backendStartCommand(platform, href = window.location.href){
    const url = new URL(href);
    let directory = '';
    if (url.protocol === 'file:'){
        try {
            directory = decodeURIComponent(url.pathname).replace(/\/[^/]*$/, '') || '/';
        } catch (error) {
            // Leave malformed file URLs to the user's terminal working directory.
        }
        if (platform === 'windows'){
            directory = directory.replace(/^\/([a-z]:)/i, '$1').replace(/\//g, '\\');
            if (url.hostname) directory = '\\\\' + url.hostname + directory;
        }else if (url.hostname){
            // A network file URL does not reveal its mounted path on macOS or Linux.
            directory = '';
        }
        // A command must stay on one line even if the file name contains a control character.
        if (/[\x00-\x1f\x7f]/.test(directory)) directory = '';
    }
    if (platform === 'windows'){
        const cd = directory ? "Set-Location -LiteralPath '" + directory.replace(/'/g, "''") + "' -ErrorAction Stop; " : '';
        return cd + 'python .\\web_main.py';
    }
    const cd = directory ? "cd -- '" + directory.replace(/'/g, "'\"'\"'") + "' && " : '';
    return cd + 'python3 web_main.py';
}

function updateBackendCommand(){
    const platform = document.getElementById('backend_platform').value;
    document.getElementById('backend_command').value = backendStartCommand(platform);
    document.getElementById('backend_terminal').textContent = platform === 'windows' ? 'PowerShell' : 'Terminal';
}

function setBackendStatus(status){
    if (status==='connected')
        backend_status.textContent = 'Backend: √';
    else
        backend_status.textContent = 'Backend: x';
    backend_status.classList.toggle('text-success', status === 'connected');
    backend_status.classList.toggle('text-secondary', status !== 'connected');
    backend_status.title = 'Open backend connection and startup help';
}

function showBackendPrompt(force = false){
    if (backend_prompt_disabled && !force) return;
    if (backend_modal_element.classList.contains('show')) return;
    document.getElementById('backend_title').textContent = isLocalBackend() ? 'Start the backend?' : 'Backend unavailable';
    document.getElementById('backend_local_help').hidden = !isLocalBackend();
    document.getElementById('backend_remote_help').hidden = isLocalBackend();
    document.getElementById('backend_address').textContent = host;
    backend_message.textContent = isLocalBackend() ?
        'The backend could not be reached. If it is running, check browser permissions for local network access.' :
        'The backend could not be reached. Check the server address and your connection.';
    backend_modal.show();
}

function checkBackendConnection(){
    const check = document.getElementById('backend_check');
    check.disabled = true;
    connectSocket(() => {
        check.disabled = false;
        backend_message.textContent = 'Connected. Close this dialog and retry your operation.';
    }, () => {
        check.disabled = false;
        backend_message.textContent = isLocalBackend() ?
            'Still unable to connect. Start the backend and allow local network access if your browser asks. This dialog will check again automatically.' :
            'Still unable to connect. Check your connection or contact the server operator. This dialog will check again automatically.';
    });
}

async function copyBackendCommand(){
    const command = document.getElementById('backend_command');
    try {
        await navigator.clipboard.writeText(command.value);
        backend_message.textContent = 'Command copied. Paste it into ' + document.getElementById('backend_terminal').textContent + ' and press Enter.';
    } catch (error) {
        // Clipboard permissions vary for file URLs; leave the command selected for manual copying.
        command.focus();
        command.select();
        backend_message.textContent = 'Select and copy the command, then paste it into your terminal and press Enter.';
    }
}

function handleBackendError(error){
    if (axios.isAxiosError(error) && !error.response){
        changeNumOfSOS(0);
        setBackendStatus('offline');
        showBackendPrompt();
    }else{
        setBackendStatus('request failed');
        backend_status.title = error.response ?
            'The backend returned HTTP ' + error.response.status + '. Check its terminal for details.' :
            'Unable to process the backend response. Check the browser console for details.';
        console.error(error);
    }
}

document.getElementById('backend_platform').value = /Windows/i.test(navigator.userAgent) ? 'windows' : 'posix';
updateBackendCommand();
document.getElementById('backend_platform').addEventListener('change', updateBackendCommand);
document.getElementById('backend_copy').addEventListener('click', copyBackendCommand);
document.getElementById('backend_check').addEventListener('click', checkBackendConnection);
backend_status.addEventListener('click', () => showBackendPrompt(true));
document.getElementById('backend_dismiss_session').addEventListener('click', () => {
    backend_prompt_disabled = true;
    try {
        sessionStorage.setItem('triples_backend_prompt_disabled', 'true');
    } catch (error) {
        // The preference still applies to the current page if storage is unavailable.
    }
    backend_modal.hide();
});
backend_modal_element.addEventListener('shown.bs.modal', () => {
    backend_poll = setInterval(() => {
        if (!document.getElementById('backend_check').disabled) checkBackendConnection();
    }, 5000);
    checkBackendConnection();
});
backend_modal_element.addEventListener('hide.bs.modal', () => {
    clearInterval(backend_poll);
    backend_poll = null;
});
