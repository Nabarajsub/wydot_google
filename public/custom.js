// Force light theme on load — override any saved dark mode preference
(function () {
    try {
        // Chainlit stores theme in localStorage as "chainlit-theme"
        localStorage.setItem("chainlit-theme", '"light"');
        // Also set the HTML attribute Chainlit uses
        document.documentElement.setAttribute("data-theme", "light");
        document.documentElement.style.colorScheme = "light";
    } catch (e) { }
})();

(function () {
    try {
        function setNativeValue(element, value) {
            const valueSetter = Object.getOwnPropertyDescriptor(element, 'value').set;
            const prototype = Object.getPrototypeOf(element);
            const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value').set;

            if (valueSetter && valueSetter !== prototypeValueSetter) {
                prototypeValueSetter.call(element, ""); // Clear first
                prototypeValueSetter.call(element, value);
            } else {
                valueSetter.call(element, ""); // Clear first
                valueSetter.call(element, value);
            }
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }

        function addCustomElements() {
            const passwordInput = document.querySelector('input[type="password"]');
            if (!passwordInput) return;

            const form = passwordInput.closest('form');
            if (!form) return;

            // 1. Add "Create an account" link
            if (!document.getElementById('custom-register-link')) {
                const div = document.createElement('div');
                div.id = 'custom-register-link';
                div.style.marginTop = '15px';
                div.style.textAlign = 'center';
                div.style.fontSize = '0.9rem';
                div.style.color = '#6b7280';

                const link = document.createElement('a');
                link.href = '/public/register.html';
                link.textContent = 'Create an account';
                link.style.color = '#2563eb';
                link.style.textDecoration = 'none';
                link.style.fontWeight = '500';
                link.style.marginLeft = '5px';

                link.onmouseover = function () { this.style.textDecoration = 'underline'; };
                link.onmouseout = function () { this.style.textDecoration = 'none'; };

                div.appendChild(document.createTextNode("Don't have an account?"));
                div.appendChild(link);

                form.parentNode.insertBefore(div, form.nextSibling);

                // Update "Login" button text
                const btn = form.querySelector('button[type="submit"]');
                if (btn) btn.textContent = "Sign In";
            }

            // 2. Add "Continue as Guest" button
            if (!document.getElementById('custom-guest-btn')) {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (!submitBtn) return;

                const guestBtn = document.createElement('button');
                guestBtn.id = 'custom-guest-btn';
                guestBtn.textContent = "Continue as Guest";
                guestBtn.type = "button";
                guestBtn.style.marginTop = '10px';
                guestBtn.style.width = '100%';
                guestBtn.style.padding = '8px';
                guestBtn.style.backgroundColor = '#f3f4f6';
                guestBtn.style.color = '#4b5563';
                guestBtn.style.border = '1px solid #d1d5db';
                guestBtn.style.borderRadius = '0.5rem';
                guestBtn.style.fontWeight = '500';
                guestBtn.style.cursor = 'pointer';
                guestBtn.style.transition = 'background-color 0.2s';
                guestBtn.style.zIndex = '9999'; // Ensure clickable

                guestBtn.onmouseover = function () { this.style.backgroundColor = '#e5e7eb'; };
                guestBtn.onmouseout = function () { this.style.backgroundColor = '#f3f4f6'; };

                guestBtn.onclick = function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log("Guest button clicked");

                    const form = guestBtn.closest('form');
                    if (!form) {
                        alert("Error: Form not found. Please refresh.");
                        return;
                    }

                    // Try to find inputs within THIS form
                    const passInput = form.querySelector('input[type="password"]');

                    // Robust email finder: check specific names, types, or fallback to first text input
                    const emailInput = form.querySelector('input[name="identifier"]') ||
                        form.querySelector('input[name="username"]') ||
                        form.querySelector('input[type="email"]') ||
                        Array.from(form.querySelectorAll('input')).find(i =>
                            i.type !== 'password' && i.type !== 'hidden' && i.type !== 'submit' && i.type !== 'button'
                        );

                    if (emailInput && passInput) {
                        guestBtn.textContent = "Logging in...";
                        guestBtn.disabled = true;

                        // Fill credentials
                        setNativeValue(emailInput, "guest@app.local");
                        setNativeValue(passInput, "guest");

                        // Force enable submit button just in case
                        if (submitBtn.disabled) {
                            submitBtn.disabled = false;
                            submitBtn.removeAttribute('disabled');
                        }

                        // Delay click to allow React state update
                        setTimeout(() => {
                            console.log("Simulating submit click for: " + emailInput.value);
                            submitBtn.click();

                            // Reset button in case login fails
                            setTimeout(() => {
                                if (guestBtn) {
                                    guestBtn.textContent = "Continue as Guest";
                                    guestBtn.disabled = false;
                                }
                            }, 3000);
                        }, 500); // Increased delay
                    } else {
                        console.error("Inputs not found", { form, emailInput, passInput });
                        // alert("Debug: Email input found? " + !!emailInput + ", Pass found? " + !!passInput);
                        alert("Error: Login fields not found. Please refresh.");
                    }
                };

                submitBtn.parentNode.insertBefore(guestBtn, submitBtn.nextSibling);
            }
        }

        // Init
        addCustomElements();
        const observer = new MutationObserver((mutations) => {
            addCustomElements();
        });
        observer.observe(document.body, { childList: true, subtree: true });

    } catch (err) {
        console.error("Check custom.js error:", err);
    }
})();
