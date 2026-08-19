import { useState, useEffect } from "react";
import { 
  Brain, 
  Activity, 
  Menu, 
  X, 
  Home, 
  Beaker, 
  FileText, 
  Upload, 
  Github, 
  ExternalLink,
  Settings,
  HelpCircle,
  Database,
  User,
  Atom
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { apiClient } from "@/lib/apiClient";

export function Navbar({ activeTab = "single", onTabChange }) {
  const [apiStatus, setApiStatus] = useState("unknown");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Check API health
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await apiClient.getHealth();
        setApiStatus("healthy");
      } catch (error) {
        setApiStatus("unhealthy");
      }
    };
    
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case "healthy":
        return "bg-green-500";
      case "degraded":
        return "bg-yellow-500";
      case "unhealthy":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case "healthy":
        return "Online";
      case "degraded":
        return "Degraded";
      case "unhealthy":
        return "Offline";
      default:
        return "Unknown";
    }
  };

  const navigationItems = [
    {
      id: "single",
      label: "Single Molecule",
      icon: Beaker,
    },
    {
      id: "batch",
      label: "Batch Prediction",
      icon: Database,
    },
    {
      id: "file",
      label: "File Upload",
      icon: Upload,
    },
    {
      id: "about",
      label: "About Us",
      icon: Atom ,
    }
  ];

  const NavItem = ({ item, isMobile = false }) => {
    const Icon = item.icon;
    const isActive = activeTab === item.id;
    
    return (
      <button
        onClick={() => {
          onTabChange?.(item.id);
          if (isMobile) setMobileMenuOpen(false);
        }}
        className={`
          flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors
          ${isActive 
            ? 'bg-primary text-primary-foreground' 
            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
          }
          ${isMobile ? 'w-full justify-start' : ''}
        `}
      >
        <Icon className="h-4 w-4" />
        <span>{item.label}</span>
      </button>
    );
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        
        {/* Logo and Brand */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center w-10 h-10 bg-primary rounded-lg">
              <Brain className="h-6 w-6 text-primary-foreground" />
            </div>
            <div className="hidden sm:block">
              {/* <h1 className="text-lg font-bold text-foreground">BBB Predictor</h1> */}
            </div>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center space-x-2">
            {navigationItems.map((item) => (
              <NavItem key={item.id} item={item} />
            ))}
          </nav>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center gap-2">
          
          {/* API Status */}
          <div className="hidden sm:flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(apiStatus)} animate-pulse`}></div>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
            <Badge variant="outline" className="text-xs">
              {getStatusText(apiStatus)}
            </Badge>
          </div>

          {/* More Options Dropdown */}
          {/* <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>Tools & Resources</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <a 
                  href="https://lgbm-bbb.onrender.com/docs" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  API Documentation
                  <ExternalLink className="h-3 w-3 ml-auto" />
                </a>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <a 
                  href="https://github.com/Rajnishphe/LGBM-BBB" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-2"
                >
                  <Github className="h-4 w-4" />
                  Source Code
                  <ExternalLink className="h-3 w-3 ml-auto" />
                </a>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <HelpCircle className="h-4 w-4 mr-2" />
                Help & Support
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground">
                API Status: {getStatusText(apiStatus)}
              </DropdownMenuLabel>
            </DropdownMenuContent>
          </DropdownMenu> */}

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Mobile Menu */}
          <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
            <SheetTrigger asChild className="lg:hidden">
              <Button variant="ghost" size="sm">
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80">
              <SheetHeader>
                {/* <SheetTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  BBB Predictor
                </SheetTitle> */}
                {/* <SheetDescription>
                  Navigate to different prediction tools
                </SheetDescription> */}
              </SheetHeader>
              
              <div className="mt-6 space-y-4">
                {/* Navigation Items */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-muted-foreground">Prediction Tools</h3>
                  {navigationItems.map((item) => (
                    <div key={item.id} className="space-y-1">
                      <NavItem item={item} isMobile />
                      <p className="text-xs text-muted-foreground ml-6">{item.description}</p>
                    </div>
                  ))}
                </div>

                {/* API Status */}
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">API Status</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(apiStatus)}`}></div>
                      <Badge variant="outline" className="text-xs">
                        {getStatusText(apiStatus)}
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Quick Links */}
                <div className="pt-4 border-t space-y-2">
                  <h3 className="text-sm font-medium text-muted-foreground">Quick Links</h3>
                  <div className="space-y-2">
                    <Button variant="ghost" className="w-full justify-start" asChild>
                      <a 
                        href="https://lgbm-bbb.onrender.com/docs" 
                        target="_blank" 
                        rel="noopener noreferrer"
                      >
                        <FileText className="h-4 w-4 mr-2" />
                        API Docs
                        <ExternalLink className="h-3 w-3 ml-auto" />
                      </a>
                    </Button>
                    <Button variant="ghost" className="w-full justify-start" asChild>
                      <a 
                        href="https://github.com/Rajnishphe/LGBM-BBB" 
                        target="_blank" 
                        rel="noopener noreferrer"
                      >
                        <Github className="h-4 w-4 mr-2" />
                        GitHub
                        <ExternalLink className="h-3 w-3 ml-auto" />
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}