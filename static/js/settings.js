
// Settings functionality

// Settings definition dictionary - defines the structure and properties of all settings
var settingsDefinition = {children: {
    parser: {
        name: "Parser Configs",
        id: "parser-configs",
        children: {
            preservePatterns: {
                name: "Preserve Patterns",
                description: "Specify patterns to preserve during parsing",
                type: "text",
                defaultValue: "sqrt",
                settingPath: "parser.preservePatterns"
            },
            standardizeText: {
                name: "Standardize Text",
                id: "standardize-text",
                children: {
                    omitMul: {
                        name: "Omit Mul",
                        description: "Omit multiplication signs in output",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "parser.standardizeText.omitMul"
                    },
                    omitPow: {
                        name: "Omit Pow",
                        description: "Omit power signs in output",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "parser.standardizeText.omitPow"
                    }
                }
            }
        }
    },
    sos: {
        name: "SOS Configs",
        id: "sos-configs",
        children: {
            structuralSOS: {
                name: "Structural SOS",
                id: "structural-sos",
                children: {
                    useStructuralSOS: {
                        name: "Use Structural SOS",
                        description: "Enable Structural SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.structuralSOS",
                        originalCheckboxId: "setting_method_use_StructuralSOS"
                    }
                }
            },
            linearSOS: {
                name: "Linear SOS",
                id: "linear-sos",
                children: {
                    useLinearSOS: {
                        name: "Use Linear SOS",
                        description: "Enable Linear SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.linearSOS",
                        originalCheckboxId: "setting_method_use_LinearSOS"
                    }
                }
            },
            sdpSOS: {
                name: "SDP SOS",
                id: "sdp-sos",
                children: {
                    useSDPSOS: {
                        name: "Use SDP SOS",
                        description: "Enable SDP SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.sdpSOS",
                        originalCheckboxId: "setting_method_use_SDPSOS"
                    }
                }
            },
            symmetricSOS: {
                name: "Symmetric SOS",
                id: "symmetric-sos",
                children: {
                    useSymmetricSOS: {
                        name: "Use Symmetric SOS",
                        description: "Enable Symmetric SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.symmetricSOS",
                        originalCheckboxId: "setting_method_use_SymmetricSOS"
                    }
                }
            }
        }
    }
}};

// Settings data structure with default values - initialized from settingsDefinition
var settings = {
    parser: {
        preservePatterns: "sqrt",
        standardizeText: {
            omitMul: true,
            omitPow: true
        }
    },
    sos: {
        structuralSOS: true,
        linearSOS: true,
        sdpSOS: true,
        symmetricSOS: true
    }
};

// Helper function to generate a unique ID for settings controls
function generateControlId(settingPath) {
    return 'setting_' + settingPath.replace(/\./g, '_');
}

// Generate checkbox control HTML
function createCheckboxControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    return '<div class="settings_item_control">' +
           '<input type="checkbox" id="' + controlId + '" class="settings_checkbox" ' + (setting.defaultValue ? 'checked' : '') + '>' +
           '<label for="' + controlId + '">Enable</label>' +
           '</div>';
}

// Generate text control HTML
function createTextControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    return '<div class="settings_item_control">' +
           '<input type="text" id="' + controlId + '" class="settings_textbox" value="' + setting.defaultValue + '">' +
           '</div>';
}

// Generate HTML for a single setting item
function createSettingItem(setting) {
    var controlHtml = '';
    switch (setting.type) {
        case 'checkbox':
            controlHtml = createCheckboxControl(setting);
            break;
        case 'text':
            controlHtml = createTextControl(setting);
            break;
        // Add more control types here as needed (select, etc.)
    }

    return '<div class="settings_item">' +
           '<div class="settings_item_name">' + setting.name + '</div>' +
           '<div class="settings_item_description">' + setting.description + '</div>' +
           controlHtml +
           '</div>';
}

// Recursively generate settings content from definition
function generateSettingsContent(parentNode, definition, level) {
    var html = '';

    // Determine if this is a section, subsection, or setting item
    if (definition.children) {
        // Create section or subsection header
        if (level >= 0){
            var headerTag = level === 0 ? 'h2' : 'h3';
            var headerClass = level === 0 ? 'settings_section_title' : 'settings_subsection_title';

            html += '<div id="' + definition.id + '" class="settings_section">' +
                    '<' + headerTag + ' class="' + headerClass + '">' + definition.name + '</' + headerTag + '>';
        }
        // Recursively process children
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                html += generateSettingsContent(null, definition.children[key], level + 1);
            }
        }

        html += '</div>';
    } else {
        // This is a setting item
        html += createSettingItem(definition);
    }

    if (parentNode) {
        parentNode.innerHTML += html;
    }
    return html;
}

// Recursively generate settings menu from definition
function generateSettingsMenu(parentNode, definition, level) {
    var html = '';

    // Create menu item for sections and subsections
    if (definition.id) {
        var menuClass = level === 0 ? 'settings_menu_item' : 'settings_menu_item subitem';
        html += '<button class="' + menuClass + '" onclick="scrollToSection(\'' + definition.id + '\')">' + definition.name + '</button>';
    }

    // Recursively process children
    if (definition.children) {
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                html += generateSettingsMenu(null, definition.children[key], level + 1);
            }
        }
    }

    if (parentNode) {
        parentNode.innerHTML += html;
    }
    return html;
}

// Initialize all settings from the definition
function initializeSettings() {
    // Initialize settings content
    var contentContainer = document.getElementById('settings_content');
    if (contentContainer) {
        contentContainer.innerHTML = '';
        generateSettingsContent(contentContainer, settingsDefinition, -1);
    }

    // Initialize settings menu
    var menuContainer = document.getElementById('settings_sidebar');
    if (menuContainer) {
        menuContainer.innerHTML = '';
        generateSettingsMenu(menuContainer, settingsDefinition, -1);
    }

    // Set initial values for all settings
    setInitialSettingsValues(settingsDefinition);

    // Setup event listeners for dynamic elements
    setupEventListenersRecursive(settingsDefinition);
}

// Set initial values for all settings
function setInitialSettingsValues(definition) {
    // If this is a section with children, recursively process them
    if (definition.children) {
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                setInitialSettingsValues(definition.children[key]);
            }
        }
    } else if (definition.type) {
        // This is a setting item
        var controlId = generateControlId(definition.settingPath);
        var element = document.getElementById(controlId);

        if (element) {
            var value = getSettingValue(definition.settingPath);
            if (definition.type === 'checkbox') {
                element.checked = value;
            } else if (definition.type === 'text') {
                element.value = value;
            }
        }
    }
}

// Get setting value from the settings object
function getSettingValue(settingPath) {
    var pathParts = settingPath.split('.');
    var value = settings;

    for (var i = 0; i < pathParts.length; i++) {
        if (value && typeof value === 'object' && value.hasOwnProperty(pathParts[i])) {
            value = value[pathParts[i]];
        } else {
            return null;
        }
    }

    return value;
}

// Update setting value in the settings object
function updateSetting(settingPath, value) {
    var pathParts = settingPath.split('.');
    var obj = settings;

    // Navigate to the parent object
    for (var i = 0; i < pathParts.length - 1; i++) {
        if (obj && typeof obj === 'object' && obj.hasOwnProperty(pathParts[i])) {
            obj = obj[pathParts[i]];
        } else {
            return; // Path doesn't exist
        }
    }

    // Update the value
    var lastKey = pathParts[pathParts.length - 1];
    obj[lastKey] = value;

    // Log the change for debugging (future: add save functionality here)
    // console.log('Setting updated:', settingPath, '=', value);
}


// Recursively setup event listeners
function setupEventListenersRecursive(definition) {
    if (definition.children) {
        // Process children recursively
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                setupEventListenersRecursive(definition.children[key]);
            }
        }
    } else if (definition.type) {
        // This is a setting item, setup event listener
        var controlId = generateControlId(definition.settingPath);
        var element = document.getElementById(controlId);

        if (element) {
            if (definition.type === 'checkbox') {
                // Checkbox change event
                element.addEventListener('change', function() {
                    var value = this.checked;
                    updateSetting(definition.settingPath, value);
                    
                    // Sync with original checkbox if specified
                    if (definition.originalCheckboxId) {
                        var originalCheckbox = document.getElementById(definition.originalCheckboxId);
                        if (originalCheckbox) {
                            originalCheckbox.checked = value;
                        }
                    }
                });

                // Sync original checkbox changes to this one
                if (definition.originalCheckboxId) {
                    var originalCheckbox = document.getElementById(definition.originalCheckboxId);
                    if (originalCheckbox) {
                        originalCheckbox.addEventListener('change', function() {
                            element.checked = this.checked;
                            updateSetting(definition.settingPath, this.checked);
                        });
                    }
                }
            } else if (definition.type === 'text') {
                // Text input event
                element.addEventListener('input', function() {
                    updateSetting(definition.settingPath, this.value);
                });
            }
        }
    }
}

// Open settings modal
function openSettingsModal() {
    document.getElementById('shadow').hidden = false;
    document.getElementById('settings_modal').hidden = false;
}

// Close settings modal
function closeSettingsModal() {
    document.getElementById('shadow').hidden = true;
    document.getElementById('settings_modal').hidden = true;
}

// Scroll to settings section
function scrollToSection(sectionId) {
    var section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Event listeners for modal controls
function setupModalEventListeners() {
    // Open button
    var openButton = document.getElementById('open_settings_btn');
    if (openButton) {
        openButton.addEventListener('click', openSettingsModal);
    }
    
    // Close button
    var closeButton = document.getElementById('settings_close');
    if (closeButton) {
        closeButton.addEventListener('click', closeSettingsModal);
    }
    
    // Click outside to close
    var shadow = document.getElementById('shadow');
    if (shadow) {
        shadow.addEventListener('click', function(e) {
            if (e.target.id === 'shadow') {
                closeSettingsModal();
            }
        });
    }

    // Escape key to close
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !document.getElementById('settings_modal').hidden) {
            closeSettingsModal();
        }
    });
}

// Initialize settings functionality when page loads
window.addEventListener('DOMContentLoaded', function() {
    initializeSettings();
    setupModalEventListeners();
    // initializeSettings will be called when modal opens
});
