import * as React from "react"

const Button = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'default' | 'ghost' | 'outline'
  }
>(({ className, variant = 'default', ...props }, ref) => {
  const baseClasses = "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
  
  const variantClasses = {
    default: "bg-blue-600 text-white hover:bg-blue-700 h-10 px-4 py-2",
    ghost: "hover:bg-gray-100 h-10 px-4 py-2",
    outline: "border border-gray-300 bg-white hover:bg-gray-50 h-10 px-4 py-2"
  }
  
  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      ref={ref}
      {...props}
    />
  )
})
Button.displayName = "Button"

export { Button }
