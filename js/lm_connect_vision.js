import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LMConnect.Vision",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LMConnectVision" || nodeData.name === "LMConnectH3PromptFullReference") {
            const isRef = nodeData.name === "LMConnectH3PromptFullReference";
            const prefix = isRef ? "reference_image_" : "image_";
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                this.updateImageInputs();
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                if (type === 1) { // 1 means INPUT connection changed
                    this.updateImageInputs();
                }
            };
            
            nodeType.prototype.updateImageInputs = function() {
                let lastConnectedIndex = 0;
                
                // Find highest connected image index
                for (let i = 1; i <= 5; i++) {
                    const inputName = prefix + i;
                    const inputIndex = this.findInputSlot(inputName);
                    if (inputIndex !== -1 && this.inputs[inputIndex].link != null) {
                        lastConnectedIndex = i;
                    }
                }
            
                // Desired count is one more than the last connected, max 5
                const desiredCount = Math.min(5, lastConnectedIndex + 1);
            
                // Remove unconnected inputs above desiredCount
                for (let i = 5; i > desiredCount; i--) {
                    const inputName = prefix + i;
                    const inputIndex = this.findInputSlot(inputName);
                    if (inputIndex !== -1 && this.inputs[inputIndex].link == null) {
                        this.removeInput(inputIndex);
                    }
                }
            
                // Add missing inputs up to desiredCount
                for (let i = 1; i <= desiredCount; i++) {
                    const inputName = prefix + i;
                    if (this.findInputSlot(inputName) === -1) {
                        this.addInput(inputName, "IMAGE");
                    }
                }
            };
        }
    }
});
