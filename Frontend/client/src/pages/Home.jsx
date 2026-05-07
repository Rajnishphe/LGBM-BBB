import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { SingleMoleculePrediction } from "@/components/SingleMoleculePrediction";
import { BatchPrediction } from "@/components/BatchPrediction";
import { FileUpload } from "@/components/FileUpload";
import { Github, ExternalLink } from "lucide-react";
import AboutPage from "../components/AboutPage";
import { Navbar } from "../components/Navbar";

export default function Home() {
  const [activeTab, setActiveTab] = useState("single");

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
  };

  const renderContent = () => {
    switch (activeTab) {
      case "single":
        return <SingleMoleculePrediction />;
      case "batch":
        return <BatchPrediction />;
      case "file":
        return <FileUpload />;
      case "about":
        return <AboutPage />;
      default:
        return <SingleMoleculePrediction />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <Navbar activeTab={activeTab} onTabChange={handleTabChange} />
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto">
          {renderContent()}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-muted/40 mt-20">
        <div className="container mx-auto px-4 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            
            {/* About */}
            <div className="md:col-span-2">
              <h3 className="text-lg font-semibold text-foreground mb-4">About BBB Predictor</h3>
            
              <div className="text-sm text-muted-foreground space-y-2">
                <strong> For research purposes only.</strong> A powerful tool that predict BBB .
              </div>
            </div>

          
            {/* Resources */}
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-4">Resources</h3>
              <div className="space-y-3">
                <a
                  href="https://lgbm-bbb.onrender.com/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  API Documentation
                  <ExternalLink size={12} />
                </a>
                <a
                  href="https://github.com/Rajnishphe/LGBM-BBB"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  <Github size={14} />
                  Source Code
                  <ExternalLink size={12} />
                </a>            
              </div>
              
             
            </div>
          </div>
          
          {/* Bottom Bar */}
          <div className="border-t mt-1 pt-1 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-xs text-muted-foreground">
              © 2025 BBB Predictor.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}