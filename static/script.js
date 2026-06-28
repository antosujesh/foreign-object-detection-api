document.addEventListener("DOMContentLoaded", () => {
    // 1. Slider values synchronization
    const sliders = [
        { id: "weight-ssim-slider", valId: "weight-ssim-val" },
        { id: "weight-abs-slider", valId: "weight-abs-val" },
        { id: "weight-lab-slider", valId: "weight-lab-val" },
        { id: "weight-edge-slider", valId: "weight-edge-val" },
        { id: "min-area-slider", valId: "min-area-val" },
        { id: "dl-threshold-slider", valId: "dl-threshold-val" },
        { id: "min-align-slider", valId: "min-align-val" }
    ];

    sliders.forEach(slider => {
        const el = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        if (el && valEl) {
            el.addEventListener("input", () => {
                valEl.textContent = el.value;
            });
        }
    });

    // 2. Image Previews handling
    setupImagePreview("ref-files-input", "ref-preview-container", true);
    setupImagePreview("curr-file-input", "curr-preview-container", false);
    setupImagePreview("mask-file-input", "mask-preview-container", false);

    function setupImagePreview(inputId, containerId, isMulti) {
        const input = document.getElementById(inputId);
        const container = document.getElementById(containerId);
        if (!input || !container) return;

        input.addEventListener("change", () => {
            container.innerHTML = "";
            const files = input.files;
            
            if (files.length === 0) {
                container.style.display = "none";
                return;
            }

            container.style.display = isMulti ? "grid" : "block";

            // If multi-select is enabled and there are many files, show a summarized badge
            if (isMulti && files.length > 4) {
                // Show first 3 previews, then a badge with the count
                for (let i = 0; i < 3; i++) {
                    createThumbnail(files[i], container);
                }
                const countOverlay = document.createElement("div");
                countOverlay.className = "preview-count-overlay";
                countOverlay.innerHTML = `<i class="fa-solid fa-images"></i> +${files.length - 3} Ref`;
                container.appendChild(countOverlay);
            } else {
                for (let i = 0; i < files.length; i++) {
                    createThumbnail(files[i], container);
                }
            }
        });
    }

    function createThumbnail(file, parentContainer) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement("img");
            img.src = e.target.result;
            img.className = "preview-thumbnail";
            parentContainer.appendChild(img);
        };
        reader.readAsDataURL(file);
    }

    // 3. Execution / Inspection Trigger
    const inspectBtn = document.getElementById("inspect-btn");
    const loaderContainer = document.getElementById("loader-container");
    const loaderStatus = document.getElementById("loader-status");
    const resultsCard = document.getElementById("results-card");
    const badgeStatus = document.getElementById("badge-status");

    // Stat fields
    const statSimilarity = document.getElementById("stat-similarity");
    const statAlignment = document.getElementById("stat-alignment");
    const statObjects = document.getElementById("stat-objects");
    const statTime = document.getElementById("stat-time");

    // Visual fields
    const imgResult = document.getElementById("img-result");
    const imgHeatmap = document.getElementById("img-heatmap");
    const imgMask = document.getElementById("img-mask");
    const linkResult = document.getElementById("link-result");
    const linkHeatmap = document.getElementById("link-heatmap");
    const linkMask = document.getElementById("link-mask");

    // Table elements
    const tableContainer = document.getElementById("detections-table-container");
    const tableBody = document.getElementById("detections-table-body");

    inspectBtn.addEventListener("click", async () => {
        const refInput = document.getElementById("ref-files-input");
        const currInput = document.getElementById("curr-file-input");
        const maskInput = document.getElementById("mask-file-input");

        // Validate uploads
        if (!refInput.files || refInput.files.length === 0) {
            showToast("Please upload at least one OK Reference Image.", "warning");
            return;
        }
        if (!currInput.files || currInput.files.length === 0) {
            showToast("Please upload the Inspection Image.", "warning");
            return;
        }

        // Show Loader & hide past results
        loaderContainer.classList.remove("hidden");
        resultsCard.classList.add("hidden");
        loaderStatus.textContent = "Loading deep neural networks and preparing images...";

        const formData = new FormData();
        
        // Append files
        for (let i = 0; i < refInput.files.length; i++) {
            formData.append("reference_images", refInput.files[i]);
        }
        formData.append("current_image", currInput.files[0]);
        
        if (maskInput.files && maskInput.files.length > 0) {
            formData.append("ignore_mask", maskInput.files[0]);
        }

        // Append slider overrides
        formData.append("weight_ssim", document.getElementById("weight-ssim-slider").value);
        formData.append("weight_abs_diff", document.getElementById("weight-abs-slider").value);
        formData.append("weight_lab_diff", document.getElementById("weight-lab-slider").value);
        formData.append("weight_edge_diff", document.getElementById("weight-edge-slider").value);
        formData.append("min_area", document.getElementById("min-area-slider").value);
        formData.append("min_align_score", document.getElementById("min-align-slider").value);
        formData.append("deep_feature_threshold", document.getElementById("dl-threshold-slider").value);
        formData.append("enable_dl_validation", document.getElementById("dl-validation-toggle").checked);
        formData.append("threshold_method", document.getElementById("threshold-method-select").value);

        // Slow updates of progress label to keep user engaged
        const statusMsgs = [
            "Aligning inspection image using feature matches...",
            "Computing structural similarities & delta-E color variations...",
            "Running class-agnostic deep learning validators...",
            "Filtering false positives & shadow regions...",
            "Drawing final visualizations..."
        ];
        
        let msgIdx = 0;
        const msgInterval = setInterval(() => {
            if (msgIdx < statusMsgs.length) {
                loaderStatus.textContent = statusMsgs[msgIdx++];
            }
        }, 1500);

        try {
            const response = await fetch("/detect-fod", {
                method: "POST",
                body: formData
            });

            clearInterval(msgInterval);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server error running detection pipeline.");
            }

            const data = await response.json();
            
            // Hide Loader
            loaderContainer.classList.add("hidden");
            resultsCard.classList.remove("hidden");

            // Handle output state based on API response
            if (data.status === "ALIGNMENT_FAILED") {
                badgeStatus.textContent = "ALIGNMENT FAILED";
                badgeStatus.className = "status-badge failed";
                
                statSimilarity.textContent = "N/A";
                statAlignment.textContent = data.alignment_score ? data.alignment_score.toFixed(3) : "0.0";
                statObjects.textContent = "0";
                statTime.textContent = `${data.processing_time_ms}ms`;

                // Hide visualization grid and tables
                document.querySelector(".visuals-grid").classList.add("hidden");
                tableContainer.classList.add("hidden");
                
                showToast("Image alignment failed. The vehicle position might have changed too much.", "danger");
            } else {
                // Success paths (FOD_DETECTED or NO_FOD)
                document.querySelector(".visuals-grid").classList.remove("hidden");
                
                // Set Badge Status
                if (data.status === "FOD_DETECTED") {
                    badgeStatus.textContent = `FOD DETECTED (${data.objects})`;
                    badgeStatus.className = "status-badge detected";
                    showToast(`Alert: Detected ${data.objects} potential Foreign Object(s).`, "danger");
                } else {
                    badgeStatus.textContent = "NO FOD";
                    badgeStatus.className = "status-badge no-fod";
                    showToast("Inspection passed: No foreign objects detected.", "success");
                }

                // Update Stats
                statSimilarity.textContent = `${data.similarity_score.toFixed(2)}%`;
                statAlignment.textContent = data.alignment_score.toFixed(3);
                statObjects.textContent = data.objects;
                statTime.textContent = `${data.processing_time_ms}ms`;

                // Update Images (append cache buster so browser reloads)
                const cacheBuster = `?t=${new Date().getTime()}`;
                
                imgResult.src = "/" + data.output_image + cacheBuster;
                linkResult.href = "/" + data.output_image + cacheBuster;

                imgHeatmap.src = "/" + data.difference_map + cacheBuster;
                linkHeatmap.href = "/" + data.difference_map + cacheBuster;

                imgMask.src = "/" + data.mask_image + cacheBuster;
                linkMask.href = "/" + data.mask_image + cacheBuster;

                // Build Table
                tableBody.innerHTML = "";
                if (data.detections && data.detections.length > 0) {
                    tableContainer.classList.remove("hidden");
                    data.detections.forEach((det, idx) => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${idx + 1}</td>
                            <td><strong>[${det.x}, ${det.y}, ${det.width}, ${det.height}]</strong></td>
                            <td>${det.area.toLocaleString()} px</td>
                            <td><span class="badge badge-accent">${(det.confidence * 100).toFixed(1)}%</span></td>
                            <td><code>${det.label}</code></td>
                        `;
                        tableBody.appendChild(tr);
                    });
                } else {
                    tableContainer.classList.add("hidden");
                }
            }
        } catch (err) {
            clearInterval(msgInterval);
            loaderContainer.classList.add("hidden");
            showToast(`Error: ${err.message}`, "danger");
            console.error(err);
        }
    });

    // 4. Toast Notification helper
    function showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        if (!container) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = "fa-circle-info";
        if (type === "success") icon = "fa-circle-check";
        if (type === "danger") icon = "fa-triangle-exclamation";
        if (type === "warning") icon = "fa-circle-exclamation";

        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);

        // Remove toast after 4.5 seconds
        setTimeout(() => {
            toast.style.animation = "slideIn 0.3s reverse forwards ease";
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 4500);
    }
});
